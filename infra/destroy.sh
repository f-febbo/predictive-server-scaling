#!/usr/bin/env bash
#
# Tear everything down and verify nothing survived.
#
# `terraform destroy` alone is not quite enough here: an Auto Scaling group can
# take a few minutes to drain, and a spot instance that launches during the
# teardown can outlive the group. So this scales both groups to zero first,
# waits, destroys, and then checks the account for stragglers.

set -euo pipefail

cd "$(dirname "$0")"

REGION="$(terraform output -raw region 2>/dev/null || echo "${AWS_REGION:-us-east-1}")"
PROJECT="$(terraform output -raw project 2>/dev/null || echo "predictive-autoscaler")"

echo "==> Scaling both fleets to zero before destroying"
for asg in $(terraform output -json asg_names 2>/dev/null | python3 -c \
    'import json,sys; print(" ".join(json.load(sys.stdin).values()))' 2>/dev/null || true); do
  echo "    $asg"
  aws autoscaling update-auto-scaling-group \
    --auto-scaling-group-name "$asg" \
    --min-size 0 --max-size 0 --desired-capacity 0 \
    --region "$REGION" 2>/dev/null || echo "    (already gone)"
done

echo "==> Waiting 60s for instances to terminate"
sleep 60

echo "==> terraform destroy"
terraform destroy -auto-approve

echo "==> Verifying nothing survived"

remaining_instances=$(aws ec2 describe-instances \
  --region "$REGION" \
  --filters "Name=tag:Project,Values=${PROJECT}" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'length(Reservations[].Instances[])' --output text 2>/dev/null || echo 0)

remaining_asgs=$(aws autoscaling describe-auto-scaling-groups \
  --region "$REGION" \
  --query "length(AutoScalingGroups[?starts_with(AutoScalingGroupName, '${PROJECT}')])" \
  --output text 2>/dev/null || echo 0)

echo "    EC2 instances tagged ${PROJECT}: ${remaining_instances}"
echo "    Auto Scaling groups:            ${remaining_asgs}"

if [ "${remaining_instances}" != "0" ] || [ "${remaining_asgs}" != "0" ]; then
  echo
  echo "!! Something is still running. Check the console before walking away."
  exit 1
fi

echo
echo "All clear. Nothing is still billing."
