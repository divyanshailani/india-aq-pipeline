-- ============================================================
-- Migration: Live Validation Observatory
-- Creates the isolated validation_ledger table
-- ============================================================

CREATE TABLE IF NOT EXISTS validation_ledger (
    id              SERIAL PRIMARY KEY,
    country         VARCHAR(2) NOT NULL,
    anchor_date     DATE NOT NULL,
    target_date     DATE NOT NULL,
    horizon         VARCHAR(10) NOT NULL,
    predicted_pm25  DOUBLE PRECISION NOT NULL,
    actual_pm25     DOUBLE PRECISION NOT NULL,
    error_delta     DOUBLE PRECISION NOT NULL,
    validated_at    TIMESTAMP DEFAULT NOW()
);

-- Idempotency rule: A specific country's forecast for a specific target_date
-- from a specific horizon should only have ONE validation record.
CREATE UNIQUE INDEX IF NOT EXISTS idx_validation_unique
    ON validation_ledger (country, target_date, horizon);

-- Fast lookups for aggregations
CREATE INDEX IF NOT EXISTS idx_validation_target
    ON validation_ledger (target_date);
