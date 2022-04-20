echo "Retrieving latest changes"
git pull

git add -A

git commit -am "$1"

git push -u origin master
