-- ============================================================
-- Global AQ Intelligence Platform — Database Schema (v5)
-- Database: indiaaq
-- Run: psql -U postgres -d indiaaq -f sql/schema.sql
--
-- This schema matches the v5 production database:
--   - Multi-country support (IN, US, GB, AU)
--   - NASA POWER weather columns
--   - NASA FIRMS fire data
--   - Pipeline run tracking
--   - Prediction validation logging
-- ============================================================


-- 1. STATIONS — master list of monitoring stations
CREATE TABLE IF NOT EXISTS stations (
    id              SERIAL PRIMARY KEY,
    openaq_id       INTEGER UNIQUE NOT NULL,        -- OpenAQ's location ID
    name            TEXT NOT NULL,                    -- e.g., "Punjabi Bagh Delhi"
    city            TEXT,
    state           TEXT,
    country         TEXT DEFAULT 'IN',
    country_code    CHAR(2) DEFAULT 'IN',            -- ISO 3166-1: IN, US, GB, AU
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    is_active       BOOLEAN DEFAULT true,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);


-- 2. RAW MEASUREMENTS — exactly as received from OpenAQ API
CREATE TABLE IF NOT EXISTS raw_measurements (
    id              BIGSERIAL PRIMARY KEY,
    station_id      INTEGER REFERENCES stations(id),
    sensor_id       INTEGER,
    parameter       TEXT NOT NULL,                    -- pm25, pm10, no2, co, etc.
    value           DOUBLE PRECISION,
    unit            TEXT,                             -- µg/m³, ppb, °C, etc.
    datetime_utc    TIMESTAMPTZ NOT NULL,
    datetime_local  TIMESTAMPTZ NOT NULL,
    ingested_at     TIMESTAMPTZ DEFAULT now()
);

-- Prevent duplicate readings for same station + parameter + time
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_unique
    ON raw_measurements (station_id, parameter, datetime_utc);

-- Fast lookups by station and time range
CREATE INDEX IF NOT EXISTS idx_raw_station_time
    ON raw_measurements (station_id, datetime_utc);

-- Fast lookups by parameter
CREATE INDEX IF NOT EXISTS idx_raw_parameter
    ON raw_measurements (parameter);


-- 3. CLEAN MEASUREMENTS — after 5-phase cleaning pipeline
CREATE TABLE IF NOT EXISTS clean_measurements (
    id              BIGSERIAL PRIMARY KEY,
    station_id      INTEGER REFERENCES stations(id),
    sensor_id       INTEGER,
    parameter       TEXT NOT NULL,
    value           DOUBLE PRECISION,
    unit            TEXT,
    datetime_utc    TIMESTAMPTZ NOT NULL,
    datetime_local  TIMESTAMPTZ NOT NULL,
    -- cleaning metadata
    cleaning_flags  TEXT[],                           -- e.g., {'outlier_capped', 'placeholder_removed'}
    is_valid        BOOLEAN DEFAULT true,
    cleaned_at      TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_clean_unique
    ON clean_measurements (station_id, parameter, datetime_utc);

CREATE INDEX IF NOT EXISTS idx_clean_station_time
    ON clean_measurements (station_id, datetime_utc);

CREATE INDEX IF NOT EXISTS idx_clean_parameter
    ON clean_measurements (parameter);


-- 4. DAILY FEATURES — engineered features for ML training
--    v5: includes country_code, NASA weather, precipitation,
--    wind_direction, and fire_count columns
CREATE TABLE IF NOT EXISTS daily_features (
    date            DATE NOT NULL,
    station_id      INTEGER REFERENCES stations(id),
    parameter       TEXT NOT NULL,                    -- pm25 or pm10 (target parameter)
    country_code    CHAR(2),                          -- ISO 3166-1 (v5: multi-country)
    -- aggregated daily value
    value           DOUBLE PRECISION,
    -- time features
    month           SMALLINT,
    day_of_week     SMALLINT,                         -- 0=Mon, 6=Sun
    is_weekend      BOOLEAN,
    day_of_year     SMALLINT,
    -- lag features (previous days' values)
    lag_1           DOUBLE PRECISION,
    lag_2           DOUBLE PRECISION,
    lag_3           DOUBLE PRECISION,
    lag_7           DOUBLE PRECISION,
    -- rolling features (SHIFTED: use only past values, no leakage)
    roll_3_mean     DOUBLE PRECISION,                 -- mean(lag_1, lag_2, lag_3)
    roll_7_mean     DOUBLE PRECISION,                 -- mean(lag_1..lag_7)
    roll_3_std      DOUBLE PRECISION,                 -- stddev(lag_1, lag_2, lag_3)
    -- cross-parameter features (other pollutants as inputs)
    temperature     DOUBLE PRECISION,                 -- OpenAQ sensor temperature
    humidity        DOUBLE PRECISION,                 -- OpenAQ sensor humidity
    wind_speed      DOUBLE PRECISION,                 -- OpenAQ sensor wind speed
    no2_value       DOUBLE PRECISION,
    co_value        DOUBLE PRECISION,
    o3_value        DOUBLE PRECISION,
    so2_value       DOUBLE PRECISION,
    -- NASA POWER satellite weather (v5)
    nasa_temperature    DOUBLE PRECISION,             -- T2M: 2m air temperature (°C)
    nasa_humidity       DOUBLE PRECISION,             -- RH2M: 2m relative humidity (%)
    nasa_wind_speed     DOUBLE PRECISION,             -- WS10M: 10m wind speed (m/s)
    precipitation       DOUBLE PRECISION,             -- PRECTOTCORR: corrected precip (mm/day)
    wind_direction      DOUBLE PRECISION,             -- WD10M: 10m wind direction (degrees)
    -- NASA FIRMS fire data (v5)
    fire_count          INTEGER DEFAULT 0,            -- fire hotspots within 50km radius
    -- metadata
    created_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (date, station_id, parameter)
);

CREATE INDEX IF NOT EXISTS idx_features_station
    ON daily_features (station_id, date);

CREATE INDEX IF NOT EXISTS idx_features_country
    ON daily_features (country_code, date);


-- 5. PREDICTIONS — model forecasts stored for serving
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    date            DATE NOT NULL,                    -- the date being predicted
    station_id      INTEGER REFERENCES stations(id),
    parameter       TEXT NOT NULL,                    -- pm25 or pm10
    horizon_days    SMALLINT NOT NULL,                -- 1, 3, or 7
    predicted_value DOUBLE PRECISION NOT NULL,
    lower_bound     DOUBLE PRECISION,                 -- confidence interval
    upper_bound     DOUBLE PRECISION,
    naqi_index      SMALLINT,                         -- computed NAQI score
    naqi_category   TEXT,                              -- Good/Moderate/Poor/Very Poor/Severe
    model_version   TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_predictions_lookup
    ON predictions (station_id, parameter, date, horizon_days);


-- 6. MODEL REGISTRY — track trained models and their performance
CREATE TABLE IF NOT EXISTS model_registry (
    id              SERIAL PRIMARY KEY,
    model_name      TEXT NOT NULL,                    -- e.g., 'xgboost'
    version         TEXT NOT NULL,                    -- e.g., 'v1', 'v5'
    parameter       TEXT NOT NULL,                    -- 'pm25' or 'pm10'
    country_code    CHAR(2),                          -- v5: per-country models
    horizon_days    SMALLINT NOT NULL,                -- 1, 3, or 7
    -- performance metrics
    mae             DOUBLE PRECISION,
    rmse            DOUBLE PRECISION,
    r2              DOUBLE PRECISION,
    -- model artifact
    artifact_path   TEXT,                             -- path to saved model file
    feature_names   TEXT[],                           -- ordered list of features used
    -- status
    is_active       BOOLEAN DEFAULT false,            -- only one active per param+horizon+country
    trained_at      TIMESTAMPTZ DEFAULT now(),
    notes           TEXT
);

-- Verify: only one active model per country + parameter + horizon
CREATE UNIQUE INDEX IF NOT EXISTS idx_active_model
    ON model_registry (country_code, parameter, horizon_days)
    WHERE is_active = true;


-- 7. PIPELINE RUNS — track daily pipeline executions (v5)
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              SERIAL PRIMARY KEY,
    run_id          UUID UNIQUE NOT NULL,
    run_date        TIMESTAMP NOT NULL DEFAULT now(),
    data_from       DATE,
    data_to         DATE,
    days_fetched    INTEGER DEFAULT 0,
    predictions_made INTEGER DEFAULT 0,
    validations_done INTEGER DEFAULT 0,
    live_mae        DOUBLE PRECISION,                 -- true forward validation only
    live_r2         DOUBLE PRECISION,                 -- hidden until sample count is trustworthy
    backtest_mae    DOUBLE PRECISION,                 -- recent one-step backtest
    backtest_r2     DOUBLE PRECISION,
    metric_source   TEXT,                             -- 'live', 'backtest', or 'none'
    metric_sample_count INTEGER DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'running',
    log             TEXT
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_date
    ON pipeline_runs (run_date DESC);


-- 8. PREDICTION LOG — track accuracy of past predictions (v5)
CREATE TABLE IF NOT EXISTS prediction_log (
    id              SERIAL PRIMARY KEY,
    run_id          UUID REFERENCES pipeline_runs(run_id),
    run_date        DATE NOT NULL DEFAULT CURRENT_DATE,
    country_code    CHAR(2),
    station_id      INTEGER REFERENCES stations(id),
    target_date     DATE NOT NULL,                   -- date the prediction was for
    horizon_days    INTEGER NOT NULL,
    predicted_value DOUBLE PRECISION,
    actual_value    DOUBLE PRECISION,                 -- filled by validator after actual arrives
    error           DOUBLE PRECISION,                 -- actual - predicted
    validated_at    TIMESTAMP,
    created_at      TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_prediction_log_dates
    ON prediction_log (target_date, country_code);

CREATE INDEX IF NOT EXISTS idx_prediction_log_run_target
    ON prediction_log (run_date, target_date, country_code);


-- 9. PREDICTION LOG ARCHIVE — invalid/debug rows removed from active metrics
CREATE TABLE IF NOT EXISTS prediction_log_archive
(LIKE prediction_log INCLUDING DEFAULTS INCLUDING CONSTRAINTS);

ALTER TABLE prediction_log_archive
    ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP DEFAULT now(),
    ADD COLUMN IF NOT EXISTS archive_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_prediction_log_archive_reason
    ON prediction_log_archive (archive_reason, archived_at);


-- Compatibility migration for existing local databases
ALTER TABLE pipeline_runs
    ADD COLUMN IF NOT EXISTS run_id UUID,
    ADD COLUMN IF NOT EXISTS data_from DATE,
    ADD COLUMN IF NOT EXISTS data_to DATE,
    ADD COLUMN IF NOT EXISTS days_fetched INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS predictions_made INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS validations_done INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS live_mae DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS live_r2 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS backtest_mae DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS backtest_r2 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS metric_source TEXT,
    ADD COLUMN IF NOT EXISTS metric_sample_count INTEGER DEFAULT 0;

ALTER TABLE prediction_log
    ADD COLUMN IF NOT EXISTS run_id UUID,
    ADD COLUMN IF NOT EXISTS run_date DATE DEFAULT CURRENT_DATE,
    ADD COLUMN IF NOT EXISTS country_code CHAR(2),
    ADD COLUMN IF NOT EXISTS station_id INTEGER REFERENCES stations(id),
    ADD COLUMN IF NOT EXISTS horizon_days INTEGER,
    ADD COLUMN IF NOT EXISTS error DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS validated_at TIMESTAMP;


-- ============================================================
-- Verify all tables created
-- ============================================================
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
