# Minimal makefile for Sphinx documentation
#
export READTHEDOCS=1

# HTML Documentation Technotes:
# - exec `make html` from docs/ to generate html documentation
# - first run will display a number of warnings (normal behavior)
# - second and subsequent runs should not display any warnings

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Clean up intermediate system-generated docs in source/ and build/
clean:
	@echo "clean $(SOURCEDIR)..."
	@rm -rf $(SOURCEDIR)/peekingduck*.rst
	@rm -rf $(SOURCEDIR)/nodes
	@echo "clean $(BUILDDIR)..."
	@rm -rf $(BUILDDIR)/doctrees
	@rm -rf $(BUILDDIR)/html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
