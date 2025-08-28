from transformers import TrainerCallback
import optuna

class OptunaPruningCallback(TrainerCallback):
    """Custom Optuna pruning callback for Hugging Face Trainer."""
    def __init__(self, trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Use the validation loss (or any other metric) for pruning
        if metrics and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            self.trial.report(eval_loss, step=state.global_step)

            # Prune the trial if needed
            if self.trial.should_prune():
                raise optuna.TrialPruned()
