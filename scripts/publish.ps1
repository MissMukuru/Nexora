param(
    [string]$Message = "Update project"
)

git add .
git commit -m $Message
git push origin main
