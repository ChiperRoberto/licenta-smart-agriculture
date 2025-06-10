.PHONY: test coverage clean

test:
	pytest --cov=.

coverage:
	coverage run -m pytest
	coverage report -m
	coverage html
	@echo "HTML report: htmlcov/index.html"

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
