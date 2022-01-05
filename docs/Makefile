# Minimal makefile for Sphinx documentation
#
export READTHEDOCS=1

# Technotes:
# - exec `make html` from docs/ folder
# - first run will display a number of warnings (normal behavior)
# - second and subsequent runs should not display any warnings

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Clean up intermediate system-generated docs for a blank slate
# NB: need master.rst, so back it up first, then restore
# clean the working build/ folder
clean:
	@echo "clean $(SOURCEDIR) (except master.rst)..."
	@mv $(SOURCEDIR)/master.rst $(SOURCEDIR)/master.bak
	@rm -rf $(SOURCEDIR)/*.rst
	@mv $(SOURCEDIR)/master.bak $(SOURCEDIR)/master.rst
	@echo "clean $(BUILDDIR)..."
	@rm -rf $(BUILDDIR)/doctrees
	@rm -rf $(BUILDDIR)/html

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
