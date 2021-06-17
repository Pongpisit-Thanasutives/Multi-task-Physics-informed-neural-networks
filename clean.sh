rm -rf ./__pycache__
rm -rf ./.ipynb_checkpoints
rm -rf ./.virtual_documents
rm -rf /Users/pongpisit/Library/Caches/pip

FILE=./kite_tutorial.ipynb
if test -f "$FILE"; then
    echo "Removing $FILE"
    rm $FILE 
fi

FILE=./.DS_store
if test -f "$FILE"; then
    echo "Removing $FILE"
    rm $FILE 
fi

FILE=./catboost_info
if [ -d "$FILE" ]; then
    echo "Removing $FILE"
    rm -rf $FILE
fi
