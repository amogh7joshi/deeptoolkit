install:
	pipenv install
upload:
	rm dist/*
	python3 setup.py sdist bdist_wheel
	twine upload dist/*
	rm -rfi deeptoolkit.egg-info
