# Release Runbook

## Staging
1. Confirm rollout exists:
   kubectl -n model-staging get rollout model

2. Confirm newest pre AnalysisRun is Successful:
   kubectl -n model-staging get analysisrun --sort-by=.metadata.creationTimestamp | tail -n 6

3. Promote:
   kubectl -n model-staging argo rollouts promote model

4. Confirm newest post AnalysisRun is Successful:
   kubectl -n model-staging get analysisrun --sort-by=.metadata.creationTimestamp | tail -n 6

## Prod (manual)
1. Confirm rollout exists:
   kubectl -n model-prod get rollout model

2. Confirm newest pre AnalysisRun is Successful:
   kubectl -n model-prod get analysisrun --sort-by=.metadata.creationTimestamp | tail -n 6

3. Promote:
   kubectl -n model-prod argo rollouts promote model

4. Confirm newest post AnalysisRun is Successful:
   kubectl -n model-prod get analysisrun --sort-by=.metadata.creationTimestamp | tail -n 6
