# MLflow Tracking Server
#   backend  : PostgreSQL (DB: mlflow)
#   artifact : S3 호환 저장소(RustFS 등) 버킷 — 서버가 프록시하므로 클라이언트는 S3 키 불필요
#   UI       : http://localhost:5000
#
# 실행: powershell -ExecutionPolicy Bypass -File scripts\start_mlflow.ps1
# 접속 정보는 프로젝트 루트의 .env (git 에 올리지 않음) 또는 환경변수로 넣는다. 형식은 .env.example 참고.

$Root = Split-Path -Parent $PSScriptRoot
$EnvFile = Join-Path $Root ".env"
if (Test-Path $EnvFile) {
    foreach ($line in Get-Content $EnvFile) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$' -and -not [Environment]::GetEnvironmentVariable($Matches[1])) {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim('"'), "Process")
        }
    }
}

function Get-Required($name) {
    $v = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrEmpty($v)) { throw "$name 이(가) 설정되지 않았습니다. .env 또는 환경변수에 넣어 주세요 (.env.example 참고)." }
    return $v
}
function Get-OrDefault($name, $default) {
    $v = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrEmpty($v)) { return $default } else { return $v }
}

$PgUri     = Get-Required "MLFLOW_BACKEND_URI"
$Artifacts = Get-OrDefault "MLFLOW_ARTIFACTS" "s3://mlflow"
$Port      = Get-OrDefault "MLFLOW_PORT" "5000"
$null = Get-Required "MLFLOW_S3_ENDPOINT_URL"
$null = Get-Required "AWS_ACCESS_KEY_ID"
$null = Get-Required "AWS_SECRET_ACCESS_KEY"
$env:AWS_DEFAULT_REGION = Get-OrDefault "AWS_DEFAULT_REGION" "us-east-1"

# mlflow.exe 대신 python -m mlflow: 폴더를 옮겨도 가상환경 실행 파일 경로가 깨지지 않음
& "$Root\.venv\Scripts\python.exe" -m mlflow server `
    --backend-store-uri $PgUri `
    --artifacts-destination $Artifacts `
    --serve-artifacts `
    --host 0.0.0.0 `
    --port $Port `
    --workers 1   # Windows 에서 uvicorn 다중 워커는 요청이 간헐적으로 멈추는 문제가 있어 1개로 고정
