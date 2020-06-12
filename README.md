# sd_rt
Version of rt.live calc for SD

### SD Zipcode ( / Docker ) version of https://github.com/k-sys/covid-19

```
git clone git@github.com:mulroony/sd_rt.git
cd sd_rt
docker build  -t sd_covid:20200611 -f Dockerfile .
docker run -v `pwd`/outputs:/covid/outputs -it sd_covid:20200611
```

 Will generate XLSX file with a sheet per zip code

#### Had to change some code from k-sys/covid-19 package. Has not been tested. Do not trust results
