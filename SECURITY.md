# Security and privacy notes

This service processes resumes, job descriptions, generated application materials,
and application history. Treat all of them as personal or confidential data.

## Required deployment controls

- Keep `.env` outside source control and container images.
- Rotate any credential that appears in terminal, CI, or tool logs.
- Configure `API_TOKEN` for every non-local deployment.
- Restrict `CORS_ORIGINS` to the actual frontend origins.
- Terminate TLS at the ingress or reverse proxy.
- Mount persistent data volumes with least-privilege filesystem permissions.
- Do not log resume text, generated materials, API tokens, or model-provider payloads.
- Define retention and deletion periods for resumes, Chroma indexes, application
  history, generated files, and evaluation exports.
- Require an explicit human confirmation before recording an application as
  submitted or connecting the service to an external application platform.

## Secret rotation procedure

1. Revoke the exposed provider credential in the provider console.
2. Create a replacement credential with the minimum required scope.
3. Update only the local or deployment secret store; do not paste it into source.
4. Restart the affected services and verify `/ready` without printing environment
   variables or rendering the fully resolved Compose configuration.
5. Review accessible terminal and CI logs according to the platform's retention
   policy.

## Evaluation-data boundary

The public-job snapshots in `data/eval_dataset/job_agent_v1` are historical
evaluation fixtures. Candidate profiles are explicitly synthetic. Do not mix real
candidate data into the checked-in evaluation set. Human annotations over private
resumes should use pseudonymous candidate identifiers and an access-controlled
annotation store.
