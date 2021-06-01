DISCORD_WEBHOOK=$1
if [[ $(TZ='Singapore' date --date="yesterday" +%u) -gt 5 ]];
then
	date_filter=$(TZ='Singapore' date --date="3 days ago" +"%Y-%m-%d")
else
	date_filter=$(TZ='Singapore' date --date="yesterday" +"%Y-%m-%d")
fi

echo "Getting issues after $date_filter"

open_issues=$(gh issue list --state all --search "created:>=$date_filter"  --json number,title,url)

merged_pr=$(gh pr list --state merged --search "merged:>=$date_filter" --json number,title,url,mergedBy,mergedAt)

opened_pr=$(gh pr list --state open --search "created:>=$date_filter" --json number,title,url)

python ./scripts/webhook_issues.py "$open_issues" "$merged_pr" "$opened_pr" "$date_filter" "$DISCORD_WEBHOOK"

