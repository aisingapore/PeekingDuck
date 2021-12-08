VERSION=$1
PLACEHOLDER='__version__ = "developer"'
VERSION_FILE='peekingduck/__init__.py'
echo "Publishing version $VERSION to PyPI..."
if grep "$PLACEHOLDER" "$VERSION_FILE"; then
	sed -i "s/$PLACEHOLDER/__version__ = \"${VERSION}\"/g" "$VERSION_FILE"
	echo "changed version to ${VERSION}"
else
	echo "not found"
	exit 1
fi
