# Второе ДЗ: online_inference

## Основные команды

Install
~~~
git clone https://github.com/made-ml-in-prod-2021/Ilya2567.git
cd Ilya2567/
git checkout homework2
cd online_inference/

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

Build docker:
~~~
docker build -t ilya4678/model_inference:v1 .
~~~

Run docker and make requests  
~~~
docker run --network host ilya4678/model_inference:v1

python -m src.query
~~~

Push docker
~~~
docker login --username ilya4678
docker tag model_inference:v1 ilya4678/model_inference:v1
docker push ilya4678/model_inference:v1
~~~

Pull docker
~~~
docker pull ilya4678/model_inference:v1
docker run --network host ilya4678/model_inference:v1
~~~

## Самооценка

(1) Оберните inference вашей модели в rest сервис.
<br>Баллы: +3 балла

(3) Напишите скрипт, который будет делать запросы к вашему сервису.
<br>Баллы: +2 балла

(4) Сделайте валидацию входных данных.
<br>Баллы: +3 балла

(5) Напишите dockerfile, соберите на его основе образ и запустите локально контейнер(docker build, docker run).
<br>Баллы: +4 балла

(6) Оптимизируйте размер docker image.
- Использовал slim версию питона (1.38G -> 607M)
- Создал отдельный файл `requirements_docker.txt`, с м**е**ньшим количеством пакетов 
- Создал файл .dockerignore, чтобы исключить копирование служебных файлов PyCharm

<br>Баллы: +3 балла

(7) опубликуйте образ в https://hub.docker.com/.
<br>Команды указаны выше.
<br>Баллы: +2 балла

(8) напишите в readme корректные команды docker pull/run, которые должны привести к тому, что локально поднимется на inference ваша модель.
<br>Команды указаны выше.
<br>Баллы: +1 балл

(5) проведите самооценку.
<br>Баллы: +1 балл

Итого: 19 баллов