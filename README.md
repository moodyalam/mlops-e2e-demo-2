# github-lab-ci (revised)

A small machine learning web service used in the OST lab *Building an End-to-End Testing Pipeline with GitHub Actions*. It trains a scikit-learn model on the Iris dataset, serves it with FastAPI, and is tested and built automatically by a GitHub Actions workflow.

![CI](https://github.com/moodyalam/mlops-e2e-demo/actions/workflows/ci.yml/badge.svg)

> Students: after forking, replace `moodyalam/mlops-e2e-demo` in the badge address above with your own `username/repo`, so the badge shows the status of *your* pipeline.

## Project layout

```
app.py                  FastAPI service: /health and /predict
model.py                trains the model
requirements.txt        packages the service needs (used by Docker)
requirements-dev.txt    the above plus the testing tools
Dockerfile              packages the service as a container image
pytest.ini              test settings
tests/
  test_1_unit.py        unit tests (model only)
  test_2_integration.py integration tests (API + model, in memory)
  test_3_e2e.py         end-to-end tests (real HTTP, real server or container)
.github/workflows/ci.yml   the CI pipeline
```

## Quick start

Needs Python 3.12 (or newer) and Git. Docker is optional for running locally.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
pytest                            # runs all tests
uvicorn app:app --reload --port 8000    # start the service, then open http://127.0.0.1:8000/docs
```

Docker (optional):

```bash
docker build -t mlapp .
docker run -d --name mlapp -p 8000:8000 mlapp
curl http://127.0.0.1:8000/health
docker stop mlapp && docker rm mlapp
```

The full lab instructions are in the accompanying notebook.
