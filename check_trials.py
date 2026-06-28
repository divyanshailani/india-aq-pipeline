import modal
import optuna

app = modal.App("check-trials")
volume = modal.Volume.from_name("pow-v12-storage")

@app.function(volumes={"/storage": volume})
def get_trials():
    db_path = "/storage/optuna_dbs/optuna_v12_US_1d.db"
    try:
        study = optuna.load_study(study_name="optuna_v12_US_1d", storage=f"sqlite:///{db_path}")
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print("\n" + "="*50)
        print(f"🔥 LIVE PROGRESS (US 1d): {completed} / 150 Trials Completed! 🔥")
        print("="*50 + "\n")
    except Exception as e:
        print(f"Error reading DB: {e}")

@app.local_entrypoint()
def main():
    get_trials.remote()
