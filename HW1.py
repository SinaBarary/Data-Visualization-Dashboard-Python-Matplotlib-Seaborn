import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

### Linear Chart ###
"""
file_path = "D:\\DataMining\\DataSets\\AirQualityUCI.csv"

# بارگیری دیتاست
air_quality = pd.read_csv(file_path, sep=";", decimal=",", dayfirst=True)

# پردازش داده‌ها
air_quality["Date"] = pd.to_datetime(air_quality["Date"], dayfirst=True)
air_quality["Time"] = pd.to_timedelta(air_quality["Time"] + ":00")  # اضافه کردن ثانیه به مقادیر زمان
air_quality["Datetime"] = air_quality["Date"] + air_quality["Time"]
air_quality = air_quality.drop(columns=["Date", "Time"])
air_quality = air_quality.rename(columns={"CO(GT)": "CO", "PT08.S1(CO)": "PT08CO"})

#رسم یک نمودار براساس کل بازه زمانی داده ها که خروجی مناسبی نبود

# پردازش داده‌ها
air_quality["Date"] = pd.to_datetime(air_quality["Date"], dayfirst=True)
air_quality["Time"] = pd.to_timedelta(air_quality["Time"] + ":00")  # اضافه کردن ثانیه به مقادیر زمان
air_quality["Datetime"] = air_quality["Date"] + air_quality["Time"]
air_quality = air_quality.drop(columns=["Date", "Time"])
air_quality = air_quality.rename(columns={"CO(GT)": "CO", "PT08.S1(CO)": "PT08CO"})

# رسم نمودار خطی
plt.figure(figsize=(10, 6))
plt.plot(air_quality["Datetime"], air_quality["CO"], label="CO levels")
plt.title("CO Levels Over Time")
plt.xlabel("Time")
plt.ylabel("CO Level")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

#رسم چند نمودار خطی 

# جداسازی داده‌ها بر اساس ماه
air_quality["Month"] = air_quality["Datetime"].dt.month

# رسم نمودار برای هر ماه
months = air_quality["Month"].unique()

for i, month in enumerate(months, start=1):
    month_data = air_quality[air_quality["Month"] == month]
    plt.figure(figsize=(15, 8))
    plt.plot(month_data["Datetime"], month_data["CO"], label="CO levels")
    plt.title(f"CO Levels for Month {month}")
    plt.xlabel("Time")
    plt.ylabel("CO Level")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()    
"""

### Scatter Plot ###
"""
# بارگیری داده‌های Iris
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris_data = pd.read_csv(url, header=None, names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])

# رسم نمودار پراکندگی
plt.figure(figsize=(8, 6))
plt.scatter(iris_data['SepalLengthCm'], iris_data['SepalWidthCm'], label='Sepal', color='blue')
plt.scatter(iris_data['PetalLengthCm'], iris_data['PetalWidthCm'], label='Petal', color='red')
plt.xlabel('Length (cm)')
plt.ylabel('Width (cm)')
plt.title('Scatter Plot of Sepal and Petal Dimensions')
plt.legend()
plt.grid(True)
plt.show()
"""

### Bar Chart ###
"""
# بارگیری داده‌ها
vaccination_data = pd.read_csv('D:\\DataMining\\DataSets\\country_vaccinations.csv')

# تعداد واکسینه‌زده‌شدگان بر اساس هر کشور
country_vaccination_counts = vaccination_data.groupby('country')['total_vaccinations'].max().sort_values(ascending=False).head(10)

# رسم نمودار میله‌ای
plt.figure(figsize=(10, 6))
country_vaccination_counts.plot(kind='bar', color='green')
plt.title('Top 10 Countries by Total Vaccinations')
plt.xlabel('Country')
plt.ylabel('Total Vaccinations')
plt.xticks(rotation=45)
plt.show()
"""

### Pie Chart ###
"""
# بارگیری داده‌ها
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv(url, header=None, names=column_names)

# رسم نمودار دایره‌ای
plt.figure(figsize=(8, 8))
iris_data['class'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue', 'lightgreen'])
plt.title('Iris Flower Species Distribution (Pie Chart)')
plt.ylabel('')
plt.show()
"""

### Bubble Chart ###
"""
# خواندن داده‌ها
data_path = "D:\\DataMining\\DataSets\\world-happiness-report-2019.csv"
data = pd.read_csv(data_path)

# ایجاد نمودار حبابی
plt.figure(figsize=(10,6))

# نمودار حبابی
plt.scatter(data['Log_of_GDP_per_capita'], data['Healthy_life_expectancy'], s=data['Social_support'] * 100, alpha=0.5)

# تنظیمات دیگر
plt.xlabel('Log of GDP per capita')
plt.ylabel('Healthy life expectancy')
plt.title('World Happiness Report 2019 - Bubble Chart')
plt.grid(True)

plt.show()
"""

### Sankey Diagram ###
"""
from matplotlib.sankey import Sankey

# بارگذاری داده
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# تقسیم داده به دو دسته بر اساس کیفیت
data['quality_label'] = data['quality'].apply(lambda x: 'bad' if x <= 5 else 'good')
quality_counts = data['quality_label'].value_counts()

# رسم نمودار Sankey
sankey = Sankey(flows=[quality_counts['bad'], quality_counts['good']],
                labels=['Bad Quality', 'Good Quality'],
                orientations=[0, 0])

sankey.finish()
plt.title('Wine Quality Sankey Diagram')
plt.show()
"""

### TreeMap Chart ###
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# بارگیری داده‌های Iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# ساخت مدل درخت تصمیم
clf = DecisionTreeClassifier()
clf.fit(data.iloc[:, :-1], data.iloc[:, -1])

# رسم درخت تصمیم
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
"""

### Histogram Chart ###
"""
# خواندن دیتاست از ریپازیتوری معروف (اینجا از دیتاست iris استفاده می‌کنیم)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(url, names=column_names)

# رسم نمودار هیستوگرام برای طول گلبرگ‌ها (petal length)
plt.hist(data['petal_length'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Petal Length')
plt.ylabel('Frequency')
plt.title('Histogram of Petal Length')
plt.show()
"""

### Parallel Coordinates ###
"""
file_path = "D:\\DataMining\\DataSets\\LPR-database.csv"
LPR = pd.read_csv(file_path)

# رسم نمودار هماهنگی موازی
#sns.pairplot(LPR)
print(LPR)

# Create a Parallel Coordinates plot
fig = px.parallel_coordinates(LPR, color="Award")

# Show the plot
fig.show()

"""
### Box Plot ###
"""
# خواندن داده از فایل CSV
df = pd.read_csv('D:\\DataMining\\DataSets\\world-happiness-report-2019.csv')

# رسم نمودار جعبه‌ای برای ستون "Score"
plt.figure(figsize=(10, 6))
plt.boxplot(df['SD_of_Ladder'], vert=False)
plt.xlabel('Score')
plt.title('Boxplot of Happiness Score in World Happiness Report 2019')
plt.show()
"""

###  Heatmap ###
"""
# خواندن داده از فایل CSV
df = pd.read_csv('D:\\DataMining\\DataSets\\AirQualityUCI.csv', delimiter=';', decimal=',')



# حذف ستون‌هایی که همه‌ی مقادیر آنها NaN هستند
df = df.dropna(axis=1, how='all')
# پردازش داده‌ها
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["Time"] = pd.to_timedelta(df["Time"] + ":00")  # اضافه کردن ثانیه به مقادیر زمان
df["Datetime"] = df["Date"] + df["Time"]
df = df.drop(columns=["Date", "Time"])
df = df.rename(columns={"CO(GT)": "CO", "PT08.S1(CO)": "PT08CO"})

# رسم نمودار حرارتی
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Correlation Matrix in AirQualityUCI Dataset')
plt.show()
"""

### Network Graph ###
"""
import networkx as nx

# خواندن داده از فایل CSV
df = pd.read_csv('D:\\DataMining\\DataSets\\world-happiness-report-2019.csv')

# ساخت گراف
G = nx.from_pandas_edgelist(df, 'SD_of_Ladder', 'Ladder')

# رسم نمودار گراف
plt.figure(figsize=(10, 6))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', linewidths=1, font_size=10)
plt.title('Network Graph')
plt.show()
"""

###  Radar Chart ###
"""
import numpy as np

# خواندن داده از فایل CSV
df = pd.read_csv('D:\\DataMining\\DataSets\\world-happiness-report-2019.csv')
selected_columns = ['Positive_affect', 'Negative_affect', 'Social_support', 'Freedom', 'Corruption']

# مقادیر میانگین ستون‌های انتخاب شده را محاسبه می‌کنیم
values = df[selected_columns].mean().values

# تعداد متغیرها
num_vars = len(selected_columns)

# زاویه‌های مدار
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# اولین زاویه به عنوان آخرین زاویه اضافه می‌شود تا نمودار راداری بسته شود
values = np.concatenate((values,[values[0]]))
angles += angles[:1]


# رسم نمودار راداری
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='skyblue', alpha=0.6)
ax.plot(angles, values, color='blue', linewidth=2, linestyle='solid')
ax.set_yticklabels([])
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.show()
"""


