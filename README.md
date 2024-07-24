# crop-face

## dev

uvicorn main:app --reload

## build

docker build -t crop-face .

## run
docker run -p 8000:8000 crop-face