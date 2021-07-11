import requests
from sklearn import linear_model
from flask import Flask, redirect, url_for,session,request,render_template,session,flash
import numpy as np
import pandas as pd
from datetime import timedelta
import datetime
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import math
wc=requests.get('https://api.covid19api.com/summary')
wc=wc.json()
ind=wc['Countries'][76]
s_d=pd.read_csv('https://api.covid19india.org/csv/latest/state_wise.csv')
app=Flask(__name__)
app.secret_key='abc'
a=pd.read_csv('https://api.covid19india.org/csv/latest/state_wise_daily.csv')
c=a[::3]
deceased1 = a[2::3]
recovered1 = a[1::3]
thres = datetime.datetime(2020,3,14)
def cost(ytt,st):
    co = 0
    for i in range(len(st)):
        co = co + (st[i]-ytt[i])*(st[i]-ytt[i])
    return math.sqrt(co)
def date1(a):
    d = thres + timedelta(days = int(a[0]))
    d = str(d).split()[0]
    d = d.split("-")
    s = str(d[2])+"-"+str(d[1])+"-"+str(d[0])
    # print(s)
    return s
models = ["Random Forest Regressor","Polynomial Regressor","Decision Tree Regressor","Auto regression","Moving Average","Autoregressive Moving Average","Autoregressive Integrated Moving Average"]
def predict(st,day,x1,n):
    if n==1:
        st=(np.array(st)).reshape(-1,1)
        day=(np.array(day)).reshape(-1,1)

        #Random Forest
        tt = RandomForestRegressor()
        tt.fit(day,st)
        ytt=tt.predict(x1)
        ytt = list(ytt)
        return ytt
    if n==2:
        st=(np.array(st)).reshape(-1,1)
        day=(np.array(day)).reshape(-1,1)
        mo=PolynomialFeatures(degree=6)
        new2=mo.fit_transform(day,st)
        xtt=mo.transform(x1)
        tt=linear_model.LinearRegression()
        tt.fit(new2,st)
        ytt=tt.predict(xtt)
        ytt = list(ytt)
        return ytt
    if n==3:
        st=(np.array(st)).reshape(-1,1)
        day=(np.array(day)).reshape(-1,1)

        #Random Forest
        tt = DecisionTreeRegressor()
        tt.fit(day,st)
        ytt=tt.predict(x1)
        ytt = list(ytt)
        return ytt
    if n==4:
        # Autoregression
        model = AutoReg(st,lags = 1)
        model_fit = model.fit()
        ytt = []
        for i in range(len(x1)):
            ytt.append(model_fit.predict(i+1,i+1))
        return ytt
    if n==5:
        # Moving Average
        model = ARIMA(st, order=(0, 0, 1))
        model_fit = model.fit()
        ytt = []
        for i in range(len(x1)):
            ytt.append(model_fit.predict(i+1,i+1))
        return ytt
    if n==6:
        # Autoregressive Moving Average
        model = ARIMA(st, order=(2, 0, 1))
        model_fit = model.fit()
        ytt = []
        for i in range(len(x1)):
            ytt.append(model_fit.predict(i+1,i+1))
        return ytt
    if n==7:
        model = ARIMA(st, order=(1, 1, 1))
        model_fit = model.fit()
        ytt = []
        for i in range(len(x1)):
            ytt.append(model_fit.predict(i+1,i+1))
        return ytt

@app.route("/results")
def results():
    ytt12 = {}
    st12 = {}
    t12 = {}
    day12 = {}
    max12 = {}
    values2 = {}
    values12 = {}
    names = ['tt','tg','ap']
    for j in range(1,8):
        ytt1 = {}
        st1 = {}
        t1 = {}
        day1 = {}
        max1 = {}
        values = {}
        values1 = {}
        for name in names:
            st = list(c[name.upper()])
            # print(st)
            day = []
            for i in range(len(st)):
                day.append(i)
            x1 = np.arange(len(day)+15).reshape(-1,1)
            st = list(map(int,st))

            # Autoregression
            # model = AutoReg(st,lags = 1)
            # model_fit = model.fit()

            # Moving Average
            # model = ARIMA(st, order=(0, 0, 1))
            # model_fit = model.fit()

            # Autoregressive Moving Average
            # model = ARIMA(st, order=(2, 0, 1))
            # model_fit = model.fit()

            # Autoregressive Integrated Moving Average
            # model = ARIMA(st, order=(1, 1, 1))
            # model_fit = model.fit()
            #
            # ytt = []
            # for i in range(len(x1)):
            #     ytt.append(model_fit.predict(i+1,i+1))

            # st=(np.array(st)).reshape(-1,1)
            # day=(np.array(day)).reshape(-1,1)

            #Random Forest
            # tt = RandomForestRegressor()
            # tt.fit(day,st)

            # mo=PolynomialFeatures(degree=6)
            # new2=mo.fit_transform(day,st)
            # xtt=mo.transform(x1)
            # tt=linear_model.LinearRegression()
            # tt.fit(new2,st)
            # ytt=tt.predict(xtt)
            ytt = predict(st,day,x1,j)
            # print(j)
            # print(i)
            ytt = list(ytt)
            for i in range(len(ytt)):
                if ytt[i]<0:
                    ytt[i]=0
            t= cost(ytt,st)
            ytt1[name]=ytt
            st1[name]=st
            t1[name]=t
            max1[name]=max(max(ytt),max(st))
            day1 = x1
            values[name]=ytt[::15]
            values1[name]=st[::15]
        t12[j] = t1
        ytt12[j] = ytt1
        st12[j]=st1
        t12[j]=t1
        day12[j]=day1
        max12[j]=max1
        values2[j]=values
        values12[j]=values1
        # print(max1)
    return render_template("results.html",da=list(map(date1,x1)),labels=list(map(date1,x1))[::15],values=values2,max=max12,values1=values12,names = names,models = models,t = t12,regions={"tt":"India","tg":"Telangana","ap":"Andhra Pradesh"})
@app.route("/")
@app.route("/<name>")
def home(name=None):
    if name==None:
        name='tt'
    if name in namesDict:
        st=list(c[name.upper()])
        deceased = list(deceased1[name.upper()])
        recovered = list(recovered1[name.upper()])
        day=[]
        for i in range(len(st)):
            # st.append(c[i][name])
            day.append(i)
        day=(np.array(day)).reshape(-1,1)
        x1=(np.arange(len(day)+15)).reshape(-1,1)
        st=list(map(int,st))

        # for i in range(1,len(st)):
        #     st[i]=st[i]+st[i-1]
        # st=(np.array(st)).reshape(-1,1)
        # mo=PolynomialFeatures(degree=10)
        # new2=mo.fit_transform(day,st)
        # xtt=mo.transform(x1)
        # tt=linear_model.LinearRegression()

        # tt = RandomForestRegressor()
        # tt.fit(day,st)
        # ytt=tt.predict(x1)
        model = AutoReg(st, lags=1)
        model_fit = model.fit()
        # print(len(x1),type(x1),x1[0])
        ytt = []
        # print(model_fit.predict(1,1))
        for i in range(len(x1)):
            ytt.append(model_fit.predict(i+1,i+1))
        # print(ytt)
        # print(len(ytt),len(x1))
        for i in range(len(ytt)):
            ytt[i]=int(ytt[i])
        if name=='tt':
            wc=[ind['TotalConfirmed'],ind['TotalDeaths'],ind['TotalRecovered'],ind['NewConfirmed']]
        else:
            for i in range(len(s_d)):
                if s_d['State'][i]==s_keys[name]:
                    wc=[s_d['Confirmed'][i],s_d['Deaths'][i],s_d['Recovered'][i],st[len(c)-1]]

        #ytt=list(map(int,ytt))
        # print(type(ytt))
        ytt = list(ytt)
        for i in range(len(ytt)):
            if ytt[i]<0:
                ytt[i]=0
        t= cost(ytt,st)
        return render_template('new.html',da=list(map(date1,x1)),y=ytt,r=len(day),or1=st,l=len(x1),d1=x1,labels=list(map(date1,x1))[::15],values=ytt[::15],max=max(ytt),values1=st[::15],im=stateDict[name],sn=namesDict[name],wc=wc,p=ytt[len(c)-1],t=t,deceased = deceased[::15],dec_max = max(deceased),recovered = recovered[::15],rec_max = max(recovered),dec = deceased,rec = recovered)
    else:
        return redirect(url_for('home'))

@app.route('/predictor')
def pred():
    l = []
    for name in s_codes1:
        # print(name)
        if name in namesDict:
                st=[]
                day=[]

                for i in range(len(c[name.upper()])):
                    c1=list(c[name.upper()])
                    st.append(c1[i])
                    day.append(i+75)
                day=(np.array(day)).reshape(-1,1)
                x1=(np.arange(75,len(day)+90)).reshape(-1,1)
                st=list(map(int,st))
                for i in range(1,len(st)):
                    st[i]=st[i]+st[i-1]
                st=(np.array(st)).reshape(-1,1)
                mo=PolynomialFeatures(degree=6)
                new2=mo.fit_transform(day,st)
                xtt=mo.transform(x1)
                tt=linear_model.LinearRegression()
                tt.fit(new2,st)
                ytt=tt.predict(xtt)
                for i in range(len(ytt)):
                    ytt[i]=int(ytt[i])
                    if name == 'an':
                        l.append({"Date":date1(x1[i]),name:ytt[i][0]})
                    else:
                        l[i][name] = ytt[i][0]
    return {"states_daily":l}

s_codes1 = ['an', 'ap', 'ar', 'as', 'br', 'ch', 'ct', 'dd', 'dl', 'dn','ga', 'gj', 'hp', 'hr', 'jh', 'jk', 'ka', 'kl', 'la', 'ld', 'mh','ml', 'mn', 'mp', 'mz', 'nl', 'or', 'pb', 'py', 'rj', 'sk', 'tg', 'tn', 'tr', 'tt', 'un', 'up', 'ut', 'wb']
# stateDict={"an":"https://www.youngernation.com/wp-content/uploads/2017/12/Andaman.jpg","ap":"https://www.youngernation.com/wp-content/uploads/2017/12/Andhra-1.jpg","ar":"https://www.youngernation.com/wp-content/uploads/2017/12/Arunachal-1.jpg","as":"https://www.youngernation.com/wp-content/uploads/2017/12/Assam-1.jpg","br":"https://www.youngernation.com/wp-content/uploads/2017/12/Bihar-1.jpg","ch":"https://www.youngernation.com/wp-content/uploads/2017/12/Indian-Union-Territory-Chandigarh.jpg","ct":"https://www.youngernation.com/wp-content/uploads/2017/12/Chhattisgarh-1.jpg","dd":"https://www.youngernation.com/wp-content/uploads/2017/12/Daman-and-Diu.jpg",'dl':'https://cdn.sketchbubble.com/pub/media/catalog/product/optimized1/a/1/a1d5257e10517b286ca194ca176c63bafc47774022219aef1b2db50a81b07bd6/delhi-map-slide1.png',"dn":"https://www.youngernation.com/wp-content/uploads/2017/12/Dadra-and-Nagar-Haveli.jpg","ga":"https://www.youngernation.com/wp-content/uploads/2017/12/Goa-1.jpg","gj":"https://www.youngernation.com/wp-content/uploads/2017/12/Gujarat.jpg","hp":"https://www.youngernation.com/wp-content/uploads/2017/12/Himachal-1.jpg","hr":"https://www.youngernation.com/wp-content/uploads/2017/12/Haryana-1.jpg","jh":"https://www.youngernation.com/wp-content/uploads/2017/12/Jharkhand-1.jpg","jk":"https://thumbs.dreamstime.com/b/web-155417042.jpg","ka":"https://www.youngernation.com/wp-content/uploads/2017/12/Indian-State-Karnataka.jpg","kl":"https://www.youngernation.com/wp-content/uploads/2017/12/Kerala.jpg","la":"https://www.youngernation.com/wp-content/uploads/2017/12/Lakhadeep.jpg","ld":"https://thumbs.dreamstime.com/b/web-155417042.jpg","me":"https://www.youngernation.com/wp-content/uploads/2017/12/Meghalaya-1.jpg","mh":"https://www.youngernation.com/wp-content/uploads/2017/12/Indian-State-Maharashtra.jpg","mn":"https://www.youngernation.com/wp-content/uploads/2017/12/Manipur-1.jpg","mp":"https://www.youngernation.com/wp-content/uploads/2017/12/Indian-State-Madhya-Pradesh.jpg",'ml':'https://www.youngernation.com/wp-content/uploads/2017/12/Meghalaya-1.jpg',"mz":"https://www.youngernation.com/wp-content/uploads/2017/12/Mizoram-1.jpg","nl":"https://www.youngernation.com/wp-content/uploads/2017/12/Nagaland-1.jpg","or":"https://www.youngernation.com/wp-content/uploads/2017/12/Odisha.jpg","pb":"https://www.youngernation.com/wp-content/uploads/2017/12/Punjab-1.jpg","py":"https://www.youngernation.com/wp-content/uploads/2017/12/Puducherry.jpg","rj":"https://www.youngernation.com/wp-content/uploads/2017/12/Rajasthan.jpg","sk":"https://www.youngernation.com/wp-content/uploads/2017/12/Sikkim-1.jpg","tg":"https://www.youngernation.com/wp-content/uploads/2017/12/Telangana-1.jpg","tn":"https://www.youngernation.com/wp-content/uploads/2017/12/Indian-State-Tamil-Nadu.jpg","tr":"https://www.youngernation.com/wp-content/uploads/2017/12/Tripura-1.jpg",'tt':'https://fvmstatic.s3.amazonaws.com/maps/m/IN-EPS-02-4001.png',"up":"https://www.youngernation.com/wp-content/uploads/2017/12/Uttar-Pradesh.jpg","ut":"https://www.youngernation.com/wp-content/uploads/2017/12/Uttarakhand-1.jpg","wb":"https://www.youngernation.com/wp-content/uploads/2017/12/Indian-State-West-Bengal.jpg"}
namesDict ={'an':'Andaman And Nicobar','ap': 'Andhra Pradesh','ar' :'Arunachal Pradesh','as'  : 'Assam','br' :'Bihar','ch': 'Chandigarh','ct' :'Chattisgarh','dd': 'Daman & Diu','dl' :'Delhi','dn' : 'Dardar and Nagar Haveli','ga': 'Goa','gj' :'Gujarath','hp' :'Himachal Pradesh','hr' : 'Haryana','jh' :'Jarkhand','jk': 'Jammu And Kashmir','ka': 'Karnataka','kl': 'Kerala','ld' : 'Ladakh','la': 'Lakshwadeep','mh': 'Maharashtra','me': 'Meghalaya','ml':'Meghalaya','mn': 'Manipur','mp' : 'Madhya Pradesh','mz': 'Mizoram','nl': 'Nagaland','or': 'Orissa','pb': 'Punjab','py' :'Pondicherry','rj': 'Rajasthan','sk': 'Sikkim','tg': 'Telangana','tn': 'TamilNadu','tt':'India','tr' :'Tripura','up' :'UttarPradesh','ut': 'Uttarakand','wb' :'WestBengal','tt': 'Total'}

li=['Afghanistan', 'Albania', 'Algeria', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bolivia (Plurinational State of)', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central African Republic', 'Chad', 'Chile', 'China', 'China, Hong Kong SAR', 'China, Macao SAR', 'China, mainland', 'China, Taiwan Province of', 'Colombia', 'Congo', 'Costa Rica', "Côte d'Ivoire", 'Croatia', 'Cuba', 'Cyprus', 'Czechia', "Democratic People's Republic of Korea", 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'France', 'French Polynesia', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran (Islamic Republic of)', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Republic of Korea', 'Republic of Moldova', 'Romania', 'Russian Federation', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Sierra Leone', 'Slovakia', 'Slovenia', 'Solomon Islands', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom of Great Britain and Northern Ireland', 'United Republic of Tanzania', 'United States of America', 'Uruguay', 'Vanuatu', 'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'Yemen', 'Zambia', 'Zimbabwe']
s_names=[ 'Andaman And Nicobar', 'Andhra Pradesh', 'Arunachal Pradesh','Assam', 'Bihar', 'Chandigarh', 'Chattisgarh', 'Daman & Diu', 'Delhi','Dardar and Nagar Haveli', 'Goa', 'Gujarath', 'Himachal Pradesh','Haryana', 'Jarkhand', 'Jammu And Kashmir', 'Karnataka', 'Kerala','Ladakh', 'Lakshwadeep', 'Maharashtra', 'Meghalaya', 'Manipur','Madhya Pradesh', 'Mizoram', 'Nagaland', 'Orissa', 'Punjab','Pondicherry', 'Rajasthan', 'Sikkim', 'Telangana', 'TamilNadu','Tripura', 'UttarPradesh', 'Uttarakand', 'WestBengal', 'India']
s_keys={'an':'Andaman and Nicobar Islands', 'ap':'Andhra Pradesh', 'ar':'Arunachal Pradesh', 'as':'Assam', 'br':'Bihar', 'ch':'Chhattisgarh', 'ct':'Chandigarh', 'dd':'Daman and Diu', 'dl':'Delhi', 'dn':'Dadra and Nagar Haveli', 'ga':'Goa', 'gj':'Gujarat', 'hp':'Himachal Pradesh', 'hr':'Haryana', 'jh':'Jharkhand', 'jk':'Jammu and Kashmir', 'ka':'Karnataka', 'kl':'Kerala', 'la':'Ladakh', 'ld':'Lakshadweep', 'mh': 'Maharashtra', 'ml':'Meghalaya', 'mn':'Manipur', 'mp':'Madhya Pradesh', 'mz':'Mizoram', 'nl':'Nagaland', 'or':'Odisha', 'pb':'Punjab', 'py':'Puducherry', 'rj':'Rajasthan', 'sk':'Sikkim', 'tg':'Telangana', 'tn':'Tamil Nadu', 'tr':'Tripura','up':'Uttar Pradesh', 'ut':'Uttarakhand', 'wb':'West Bengal'}
stateDict={'an': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fan.png?alt=media',
 'ap': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fap.png?alt=media',
 'ar': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Far.png?alt=media',
 'as': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fas.png?alt=media',
 'br': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fbr.png?alt=media',
 'ch': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fch.png?alt=media',
 'ct': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fct.png?alt=media',
 'dd': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fdd.png?alt=media',
 'dl': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fdl.png?alt=media',
 'dn': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fdn.png?alt=media',
 'ga': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fga.png?alt=media',
 'gj': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fgj.png?alt=media',
 'hp': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fhp.png?alt=media',
 'hr': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fhr.png?alt=media',
 'jh': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fjh.png?alt=media',
 'jk': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fjk.png?alt=media',
 'ka': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fka.png?alt=media',
 'kl': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fkl.png?alt=media',
 'la': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fla.png?alt=media',
 'ld': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fld.png?alt=media',
 'mh': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fmh.png?alt=media',
 'ml': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fml.png?alt=media',
 'mn': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fmn.png?alt=media',
 'mp': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fmp.png?alt=media',
 'mz': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fmz.png?alt=media',
 'nl': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fnl.png?alt=media',
 'or': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2For.png?alt=media',
 'pb': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fpb.png?alt=media',
 'py': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fpy.png?alt=media',
 'rj': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Frj.png?alt=media',
 'sk': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fsk.png?alt=media',
 'tg': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Ftg.png?alt=media',
 'tn': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Ftn.png?alt=media',
 'tr': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Ftr.png?alt=media',
 'tt': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Ftt.png?alt=media',
 'up': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fup.png?alt=media',
 'ut': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fut.png?alt=media',
 'wb': 'https://firebasestorage.googleapis.com/v0/b/brave-theater-255512.appspot.com/o/states%2Fwb.png?alt=media'}
if __name__=='__main__':
	app.run(debug = True)
