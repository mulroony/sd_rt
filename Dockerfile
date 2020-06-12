FROM python:3.7.7-stretch

RUN pip install pymc3 requests openpyxl pyyaml

RUN mkdir -p /covid/ /covid/ && chmod -R 777 /covid

ADD sd_zip_rt.py /covid/

CMD ["python","/covid/sd_zip_rt.py"]
