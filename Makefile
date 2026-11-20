PY ?= python

.PHONY: install data fit evaluate decisions benchmark readme all demo test serve docker

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

data:        ## download all 15 MVTec AD categories (about 4.9 GB) from MVTec
	$(PY) scripts/01_download.py

fit:         ## PatchCore and baselines for every category, scores for every evaluation image
	$(PY) scripts/02_fit_and_score.py

evaluate:    ## AUROC with bootstrap intervals, figures
	$(PY) scripts/03_evaluate.py

decisions:   ## thresholds on calibration images, triage and cost on the holdout images
	$(PY) scripts/04_decisions.py

benchmark:   ## service latency on this machine
	$(PY) scripts/05_benchmark.py --category bottle

readme:
	$(PY) scripts/render_readme.py

all: data fit evaluate decisions benchmark readme

demo:        ## just the bottle category: download (156 MB) and fit, then serve it
	$(PY) scripts/01_download.py bottle
	$(PY) scripts/02_fit_and_score.py bottle

test:
	$(PY) -m pytest -q

serve:
	uvicorn defects.service:app --app-dir src --port 8000

docker:
	docker build -t defect-inspection .
	docker run --rm -p 8000:8000 defect-inspection
