import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.sidebar.subheader('Table of Contents')
st.sidebar.write('1. ','<a href=#case-study-on-water-potability-and-prediction-model-using-k-nearest-neighbors>Introduction</a>', unsafe_allow_html=True)
st.sidebar.write('2. ','<a href=#data-cleaning>Data cleaning</a>', unsafe_allow_html=True)
st.sidebar.write('3. ','<a href=#exploratory-data-analysis>Exploratory data analysis</a>', unsafe_allow_html=True)
st.sidebar.write('4. ','<a href=#applying-k-nearest-neighbors>K-Nearest-Neighbors machine learning model</a>', unsafe_allow_html=True)
st.sidebar.write('5. ','<a href=#interactive-prediction-of-water-potability>Interactive prediction tool</a>', unsafe_allow_html=True)
st.sidebar.write('6. ','<a href=#conclusion>Conclusion</a>', unsafe_allow_html=True)
#st.sidebar.write('2. ','<a href=></a>', unsafe_allow_html=True)
#st.sidebar.write('2. ','<a href=></a>', unsafe_allow_html=True)



st.title('Case study on water potability and prediction model using K-Nearest-Neighbors')

st.header('Goal of this case study')
st.subheader('The goal of this case study is to create a model to predict whether water is potable or not using a series of factors. ')

st.header('Definition of the water quality metrics')
st.write('The water_potability.csv file contains water quality metrics for 3276 different water bodies. Click on the expander to show all definitions of all metrics.')
with st.expander('Metric Definitions'):
    st.subheader('1. pH')
    st.write('PH is an important parameter in evaluating the acid–base balance of water. It is also the indicator of acidic or alkaline condition of water status. WHO has recommended maximum permissible limit of pH from 6.5 to 8.5. The current investigation ranges were 6.52–6.83 which are in the range of WHO standards.')
    st.subheader('2. Hardness')
    st.write('Hardness is mainly caused by calcium and magnesium salts. These salts are dissolved from geologic deposits through which water travels. The length of time water is in contact with hardness producing material helps determine how much hardness there is in raw water. Hardness was originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.')
    st.subheader('3. Solids')
    st.write('Water has the ability to dissolve a wide range of inorganic and some organic minerals or salts such as potassium, calcium, sodium, bicarbonates, chlorides, magnesium, sulfates etc. These minerals produced un-wanted taste and diluted color in appearance of water. This is the important parameter for the use of water. The water with high TDS value indicates that water is highly mineralized. Desirable limit for TDS is 500 mg/l and maximum limit is 1000 mg/l which prescribed for drinking purpose.')
    st.subheader('4. Chloramines')
    st.write('Chlorine and chloramine are the major disinfectants used in public water systems. Chloramines are most commonly formed when ammonia is added to chlorine to treat drinking water. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.')
    st.subheader('5. Sulfate')
    st.write('Sulfates are naturally occurring substances that are found in minerals, soil, and rocks. They are present in ambient air, groundwater, plants, and food. The principal commercial use of sulfate is in the chemical industry. Sulfate concentration in seawater is about 2,700 milligrams per liter (mg/L). It ranges from 3 to 30 mg/L in most freshwater supplies, although much higher concentrations (1000 mg/L) are found in some geographic locations.')
    st.subheader('6. Conductivity')
    st.write('Pure water is not a good conductor of electric current rather’s a good insulator. Increase in ions concentration enhances the electrical conductivity of water. Generally, the amount of dissolved solids in water determines the electrical conductivity. Electrical conductivity (EC) actually measures the ionic process of a solution that enables it to transmit current. According to WHO standards, EC value should not exceeded 400 μS/cm.')
    st.subheader('7. Organic Carbon')
    st.write('Total Organic Carbon (TOC) in source waters comes from decaying natural organic matter (NOM) as well as synthetic sources. TOC is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is use for treatment.')
    st.subheader('8. Trihalomethanes')
    st.write('THMs are chemicals which may be found in water treated with chlorine. The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm is considered safe in drinking water.')
    st.subheader('9. Turbidity')
    st.write('The turbidity of water depends on the quantity of solid matter present in the suspended state. It is a measure of light emitting properties of water and the test is used to indicate the quality of waste discharge with respect to colloidal matter. The mean turbidity value obtained for Wondo Genet Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU.')
    st.subheader('10. Potability')
    st.write('Indicates if water is safe for human consumption where 1 means Potable and 0 means Not potable.')

st.header('Source')
st.subheader('https://www.kaggle.com/adityakadiwal/water-potability (kaggle.com)')

st.header('Data cleaning')
st.write('All there is to do is just dropping off all the nan values, since they are not able to participate in calculations. ')
with st.echo():
    water = pd.read_csv('water_potability.csv')
    nwater = water.dropna()
st.dataframe(nwater)
st.header('Exploratory data analysis')
#column headings
columns = nwater.drop('Potability',axis=1).columns

st.subheader('Scatter Matrix')
plot2 = px.scatter_matrix(nwater,dimensions=columns, color='Potability', width=800, height=800)
st.plotly_chart(plot2)
st.write('Here is a scatterplot matrix of all the water quality metrics against each other, yellow dots are potable entries and blue dots are non-potable entries. It is clear that there is little to none separation between the blue dots and the yellow dots, therefore achieving a high accuracy with any machine learning algorithm will be difficult. ')


st.subheader('3D scatter')
col1, col2 = st.columns([1, 2])
with col1:
    xi = st.selectbox('X axis variable', columns)
    yi = st.selectbox('Y axis variable', columns)
    zi = st.selectbox('Z axis variable', columns)

xo = nwater[xi]
yo = nwater[yi]
zo = nwater[zi]
with col2:
    plot1 = px.scatter_3d(nwater,xo,yo,zo,color='Potability', width=500)
    st.plotly_chart(plot1)
st.write('Same as the one above but with 3 of the variables at a time. All of them should appear as merged blobs since increasing the number of dimensions to compare at once will not make the yellow and blue dots more separated. Use the drop down boxes on the left to select which variables to compare. It will show as a line/plane if not all 3 selected variables are different. ')

st.subheader('Correlation heatmap')
matrixwater = nwater.drop('Potability',axis=1)
plot3 = px.imshow(matrixwater.corr())
st.plotly_chart(plot3)
st.write('This correlation heat map shows the degree of correlation between each variable. 1 means the variables are perfectly linearly correlated, while 0 means the variables are perfectly uncorrelated(linearly). As seen from this heatmap and the previous scatter matrix, correlation between all variables are extremely low. This further explains the difficulty of creating a usable model for prediction. ')

st.subheader('Countplot')
barwater = nwater.groupby('Potability').count().reset_index()
plot4 = px.bar(barwater, 'Potability', 'ph', labels={'ph':'count'})
st.plotly_chart(plot4)
st.write('This countplot shows the proportion of potable waters and non-potable waters in the dataset. Roughly 60% are non-potable and 40% are potable. ')

st.header('Applying K-Nearest-Neighbors')
st.write('K-Nearest-Neighbors is a machine learning model where the prediction is made based on the distance(in terms of similarity) towards other entries in the training group, then the majority result from the closest k amounts of neighbors will be the prediction. I find this model suitable here because the other machine learning models don’t meet the requirements for this dataset. ')
st.subheader('Code of creating the model')
st.write('This is the code I used for the K-Nearest-Neighbors prediction model, click to expand the code. ')
with st.expander("Standardizing the dataset and seperating into training group and test group"):
    with st.echo():
        def standardize(series):
            """standardizes a array-like, returns array of same length"""
            return (series - np.mean(series))/np.std(series)
        #standardizes nwater
        standwater = pd.DataFrame()
        for column in columns:
            i = standardize(nwater[column])
            standwater = pd.concat([standwater,i],axis=1)
        standwater = pd.concat([standwater,nwater['Potability']],axis=1)
            
        #randomizes the seperation into train group and test group
        randwater = standwater.sample(frac=1, random_state=0).reset_index(drop=True)
            
        #seperate into train group and test group
        training_proportion = 7/9
    
        num_entries = randwater.shape[0]
        num_train = int(np.round(num_entries * training_proportion))
        num_test = num_entries - num_train
        
        #print(num_entries,num_train,num_test)
        
        trainwater = randwater.iloc[:num_train]
        testwater = randwater.iloc[num_train:num_entries]
        #print(trainwater)
        #print(testwater)
    
        #dropping potability column on both train set and test set so that amount of columns is equal for array arithmetic
        totest = testwater.drop('Potability',axis=1)
        totrain = trainwater.drop('Potability',axis=1)

with st.expander("Functions used to predict potability"):
    with st.echo():
        def distance(features1, features2):
            """The Euclidean distance between two arrays of feature values."""
            return sum((features1 - features2) ** 2) ** 0.5
    
        def most_common(label, table):
            """finds most common potability status given distance table"""
            mid = table.groupby(label).count().reset_index().rename(columns={0:'count'}).sort_values('count',ascending=False)['Potability']
            return mid.iloc[0]
    
        def alldistance(test_row, train_table):
            """calculates the distance of a row from every row in a training table, returns a table"""
            distances = []
            for i in train_table.index:
                d = distance(np.array(test_row),np.array(train_table.iloc[i]))
                distances.append(d)
                distances_table = pd.concat([trainwater['Potability'],pd.Series(distances)],axis=1)
            return distances_table
    
        def classify(test_row, train_rows, k):
            """Return the most common class among k nearest neigbors to test_row."""
            thedistancetable = alldistance(test_row, train_rows)
            thedistancetable = thedistancetable.sort_values(0, ascending = True).iloc[:k]
            return most_common('Potability', thedistancetable)

st.write('The python function classify() can now be called to predict potability, given a row of the 9 variables. The function returns a 0/1 value where 0 means non-potable and 1 means potable. ')
st.subheader('Testing the accuracy of KNN model')

codeaccuracy = """
results = []
for j in np.arange(1,100,5):
    thek = j
    tested = []
    for i in np.arange(589):
        tested.append(classify(totest.iloc[i],totrain,thek))
        results.append(sum(tested==testwater['Potability'])/len(tested))
        print(j)
        results
        
        tested= []
        for i in range(testwater.shape[0]):
            tested.append(classify(totest.iloc[i],totrain,thek))
            sum(tested==testwater['Potability'])/len(tested)
print(results)
#output
1
6
11
16
21
26
31
36
41
46
51
56
61
66
71
76
81
86
91
96
[0.6196868008948546,
 0.6756152125279642,
 0.6868008948545862,
 0.6823266219239373,
 0.6733780760626398,
 0.6756152125279642,
 0.6711409395973155,
 0.6756152125279642,
 0.6868008948545862,
 0.6778523489932886,
 0.6666666666666666,
 0.668903803131991,
 0.6666666666666666,
 0.6621923937360179,
 0.6621923937360179,
 0.6644295302013423,
 0.6599552572706935,
 0.6621923937360179,
 0.6577181208053692,
 0.6532438478747203]
            """
with st.expander("Code for testing the accuracy"):
    st.code(codeaccuracy)
    
st.write('From the above we can see that the model performs the best at k values of between 6-51, which is around 68-69%.')

st.header('Interactive prediction of water potability')
st.subheader('A simple way to utilise the prediction model is to enter values into the form below, click on the form submit button, and it will update the prediction to be whether potable or non-potable. Default values are the mean, range is from the minimum value the variable ever appeared at to maximum value the variable ever appeared at.')
colum = nwater.drop('Potability',axis=1).columns
def mandstd(a):
    meann = np.mean(a)
    stdd = np.std(a)
    minn = np.round(np.min(a))
    maxx = np.round(np.max(a))
    return meann, stdd, minn, maxx
msv = []
for h in colum:
    msv.append(mandstd(nwater[h]))
    
col3, col4, col5 = st.columns(3)
with st.form("form1"):
    with col3:    
        fa = st.number_input(colum[0],key='a', value=msv[0][0], min_value=msv[0][2], max_value=msv[0][3])
        fb = st.number_input(colum[1],key='b', value=msv[1][0], min_value=msv[1][2], max_value=msv[1][3])
        fc = st.number_input(colum[2],key='c', value=msv[2][0], min_value=msv[2][2], max_value=msv[2][3])
    with col4:
        fd = st.number_input(colum[3],key='d', value=msv[3][0], min_value=msv[3][2], max_value=msv[3][3])
        fe = st.number_input(colum[4],key='e', value=msv[4][0], min_value=msv[4][2], max_value=msv[4][3])
        ff = st.number_input(colum[5],key='f', value=msv[5][0], min_value=msv[5][2], max_value=msv[5][3])
    with col5:
        fg = st.number_input(colum[6],key='g', value=msv[6][0], min_value=msv[6][2], max_value=msv[6][3])
        fh = st.number_input(colum[7],key='h', value=msv[7][0], min_value=msv[7][2], max_value=msv[7][3])
        fi = st.number_input(colum[8],key='i', value=msv[8][0], min_value=msv[8][2], max_value=msv[8][3])
    
    fsa = fa-msv[0][0]/msv[0][1]
    fsb = fb-msv[1][0]/msv[1][1]
    fsc = fc-msv[2][0]/msv[2][1]
    fsd = fd-msv[3][0]/msv[3][1]
    fse = fe-msv[4][0]/msv[4][1]
    fsf = ff-msv[5][0]/msv[5][1]
    fsg = fg-msv[6][0]/msv[6][1]
    fsh = fh-msv[7][0]/msv[7][1]
    fsi = fi-msv[8][0]/msv[8][1]
    
    tableinput = pd.DataFrame(np.array([[fsa, fsb, fsc, fsd, fse, fsf, fsg, fsh, fsi]]), columns=colum)
    formresult = classify(tableinput.iloc[0], totrain, 25)
submitted = st.form_submit_button("Submit")
if submitted:
    formresult
    if formresult==1:
        st.write('Prediction: Potable')
    elif formresult==0:
        st.write('Prediction: Non-potable')

# standwater = pd.DataFrame()
#         for column in columns:
#             i = standardize(nwater[column])
#             standwater = pd.concat([standwater,i],axis=1)
#         standwater = pd.concat([standwater,nwater['Potability']],axis=1)
#csv i/o form
st.subheader('You could also upload a csv file of similar values to predict potability. The column order and column names must be the same as the ones I used, they were all shown above. This should return a new dataframe with an extra column of either 1/0 indicating potability.')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dfupload = pd.read_csv(uploaded_file)
    for co in colum:
        i = standardize(dfupload[co])
        standupl = pd.concat([standupl,i],axis=1)
    
    uplout= []
    for i in range(dfupload.shape[0]):
        uplout.append(classify(standupl.iloc[i],totrain,25))
    dfout = pd.concat(dfupload, pd.Series[uplout], axis=1)
    st.dataframe(dfout)
st.header('Conclusion')
st.write('Machine learning models can be fit onto datasets, but if the dataset is untrue or made up, then the model becomes meaningless. In the case of this dataset, the figures are questionable, evident from extreme values in the pH column and waters with way too much solids are yet still potable, but it is also not randomly generated since I have a accuracy greater than 50%. Some models such as logistics regression cannot be used in this case because of low correlation, so I used KNN. Furthermore, the ~69% accuracy from the testing is too low to have any real-world applications. My next step should be to use some more complex models and compare them to see which model performs the best. ')
