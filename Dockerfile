FROM python:3.7.7-stretch

RUN pip install pymc3 requests openpyxl pyyaml

RUN mkdir -p /covid/inputs /covid/ && chmod -R 777 /covid

ADD ./* /covid/inputs/

CMD ["python","/covid/inputs/sd_zip_rt.py"]
