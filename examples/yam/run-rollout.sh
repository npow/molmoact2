#!/usr/bin/env bash
# Drive the left YAM arm using a local pi0.5 Orchestral Worker, or with
# --predictive to run the local vision-guided predictive conveyor controller.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
orchestral_dir="$repo_dir/../orchestral"
robot_python="${YAM_PYTHON:-$repo_dir/.venv/bin/python}"
i2rt_dir="${I2RT_ROOT:-$repo_dir/../i2rt}"
default_grant="$orchestral_dir/.thor-pi05/grant"
rtc_config=examples/yam/configs/pi05_red_lid_conveyor_left.yaml
rtc_sdk_dir="${ORCHESTRAL_RTC_SDK_ROOT:-}"

# Managed deployments need staging credentials and a DNS override on Thor.
# Keep this path in the same launcher so local Worker usage stays unchanged.
managed=0
for arg in "$@"; do
  case "$arg" in
    --deployment|--orchestral-deployment|--soak-hours|--probe) managed=1 ;;
  esac
done
if [[ "$managed" -eq 1 ]]; then
  if [[ ! -x "$robot_python" ]]; then
    printf '%s\n' "Set YAM_PYTHON to the robot interpreter." >&2
    exit 1
  fi
  cd -- "$repo_dir"
  exec sudo -n unshare --mount -- bash -c '
    set -euo pipefail
    mount --bind "$1" "$(readlink -f /etc/resolv.conf)"
    repo="$2"; user_name="$3"; python="$4"; i2rt="$5"; credentials="$6"; shift 6
    exec runuser -u "$user_name" -- env \
      -u ORC_BASE_URL -u ORC_API_KEY \
      ORCHESTRAL_BRIDGE_IPV4=1 \
      ORC_CREDENTIALS_FILE="$credentials" \
      PYTHONPATH="$repo/examples/yam:$i2rt${PYTHONPATH:+:$PYTHONPATH}" \
      "$python" -u "$repo/examples/yam/managed_rollout.py" "$@"
  ' bash \
    "$repo_dir/examples/yam/orchestral-stage-resolv.conf" \
    "$repo_dir" "$(id -un)" "$robot_python" "$i2rt_dir" \
    "${ORC_CREDENTIALS_FILE:-$HOME/.orchestral/stage.json}" "$@"
fi

if [[ ! -x "$robot_python" ]]; then
  printf '%s\n' "Set YAM_PYTHON to the robot interpreter." >&2
  exit 1
fi

if [[ -n "$rtc_sdk_dir" && -d "$rtc_sdk_dir/src/orchestral" ]]; then
  rtc_sdk_dir="$(realpath -- "$rtc_sdk_dir")"
  export PYTHONPATH="$rtc_sdk_dir/src${PYTHONPATH:+:$PYTHONPATH}"
fi

if [[ "${1:-}" == "--gr00t" ]]; then
  shift
  gr00t_dir="$orchestral_dir/.thor-gr00t"
  export PYTHONPATH="$repo_dir/examples/yam:$i2rt_dir${PYTHONPATH:+:$PYTHONPATH}"
  cd -- "$repo_dir"
  exec "$robot_python" "$gr00t_dir/rollout.py" \
    --config-path examples/yam/configs/pi05_red_lid_conveyor_left.yaml \
    --right-config-path examples/yam/configs/yam_deployment_right.yaml \
    --orchestral-grant "${ORCHESTRAL_GRANT:-$gr00t_dir/grant}" \
    "$@"
fi

use_predictive=0
pass_args=()
prev=""
for arg in "$@"; do
  if [[ "$prev" == "--controller" ]]; then
    if [[ "$arg" == "predictive" ]]; then
      use_predictive=1
    fi
    pass_args+=("$arg")
    prev=""
    continue
  fi
  if [[ "$arg" == "--predictive" ]]; then
    use_predictive=1
    continue
  fi
  if [[ "$arg" == "--controller=predictive" ]]; then
    use_predictive=1
  fi
  pass_args+=("$arg")
  prev="$arg"
done

cd -- "$repo_dir"
export PYTHONPATH="$repo_dir/examples/yam:$i2rt_dir${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$use_predictive" -eq 1 ]]; then
  exec "$robot_python" examples/yam/predictive_conveyor_rollout.py \
    --config-path examples/yam/configs/pi05_red_lid_conveyor_left.yaml \
    --right-config-path examples/yam/configs/yam_deployment_right.yaml \
    --active-arm-side left \
    --control-hz 60 \
    "${pass_args[@]}"
fi

if [[ "${pass_args[0]:-}" == "--holesaw" ]]; then
  rtc_config=examples/yam/configs/pi05_holesaw_conveyor_left.yaml
  default_grant="$orchestral_dir/.thor-pi05/grant-holesaw"
  pass_args=("${pass_args[@]:1}")
fi
grant="${ORCHESTRAL_GRANT:-$default_grant}"

if [[ ! -s "$grant" ]]; then
  printf 'Local Worker Grant missing at %s. Start the Worker or set ORCHESTRAL_GRANT (or pass --predictive).\n' "$grant" >&2
  exit 1
fi

if [[ -z "${ORC_TELEMETRY_BASE_URL:-}" || -z "${ORC_TELEMETRY_API_KEY:-}" ]]; then
  printf '%s\n' "ORC_TELEMETRY_BASE_URL and ORC_TELEMETRY_API_KEY are not both set: the Session's control-loop spans stay on this machine." >&2
fi
grant="$(realpath -- "$grant")"

exec "$robot_python" examples/yam/rtc_rollout.py \
  --config-path "$rtc_config" \
  --right-config-path examples/yam/configs/yam_deployment_right.yaml \
  --active-arm-side left \
  --orchestral-grant "$grant" \
  --control-hz 15 \
  "${pass_args[@]}"
