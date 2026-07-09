# Lab 3.2 Policy Control

This folder is a self-contained long-form policy pipeline.

Files:

- `docs/policy_pipeline_plan.md`: overall design and ranking objective
- `scripts/longform_policy_pipeline.py`: builds the training table, trains the policy model, and runs fresh demo generations

The policy does not generate audio on its own. It learns to choose:

- which checkpoint to use
- which long-form setting preset to use

for a given song.
