# ★ House Rocket Insight's Project

![Sem título](https://user-images.githubusercontent.com/75986085/126676350-7000e56f-8fac-4fbc-b766-4c00ab2301a2.png)

<h2>Summary</h2>
<hr>

- [0. Bussiness Problem](#0-bussiness-problem)
  - [0.1. What is a Marketplace](#01-what-is-a-marketplace)
  
- [1. Solution Strategy](#1-solution-strategy)
  - [1.1. First CRISP Cycle](#11-first-crisp-cycle)

- [2. Exploratory Data Analysis](#2-exploratory-data-analysis)

<h2>0. Bussiness Problem</h2>
<hr>
<p>House Rocket bussiness model is purchasing & reselling properties located on Seattle USA on digital app..</p>
<p>The CEO of House Rocket wants to maximize the company's revenue by finding good business opportunities.
Their main strategy is to buy homes in great locations at low prices and then later resell them at higher prices. The greater the difference between buying and selling, the greater the company's profit and therefore the greater its revenue.</p>
<p>However, the houses have many examples that make our inspiration and examples and more attractive can also influence the examples. You need to answer some questions</p>

> *Which houses should the House Rocket CEO buy and at what purchase price?*
> 
> *Once the house is in the company's possession, when is the best time to sell it and what would the sale price be?*
> 
> *Should House Rocket do a renovation to raise the sale price?*
> 
> *What would be the suggestions for changes?*
> 
> *What is the increment in the price given by each reform option?*

<h3>0.1. What is a Marketplace</h3>
<p>The Marketplace-type business model, the formal definition is, basically, it brings together people who are looking for a solution of some model with this problem, for example, which is a person who is looking for a place or a residence for this problem to travel, and on the other hand, I have people who actually own a residence and are willing to allow people from other parts of the world to come and settle in that residence.</p>
<p>This example is the Marketplace model that are offered as an offer to people that are offered through Airbnb, which are examples to be chosen by Airbnb, which are examples to be chosen by Airbnb, it is an example used by Airbnb, it these people are selected with people who have an accommodation.</p>
<p>The House Rocket find people who are looking to buy houses and people who like are looking to sell a houses.</p>

<h2>1. Solution Strategy</h2>
<hr>
<h3>1.1. First CRISP Cycle.</h3>
<p>CEO requested several filters to be able to see how the company is doing!</p>

https://user-images.githubusercontent.com/75986085/164525734-eb82f6fb-c788-4113-bbf3-92b3c90fdf1a.mp4

<p>Streamlit App</p>

<ul>
  <dl>
    <dt>Data Cleaning & Descriptive Statistical.</dt>
      <dd>First real step is download the dataset, import in jupyter and start in seven steps to change data types, data dimension, fillout na... At first statistic dataframe, i used simple statistic descriptions to check how my data is organized, and check, in dataset have only numerical attirbutes!!</dd>
    <dt>Feature Engineering.</dt>
      <dd>In the first cycle i do not builded some new features, but maked some based on Date-Time for Exploratory Data Analysis.</dd>
    <dt>Data Filtering.</dt>
      <dd>Just filter simple columns "ID".</dd>
    <dt>Exploratory Data Analysis.</dt>
      <dd>This is the principal step of this project, find houses to buy, i have analised all dataset columns on three steps, univariable, bivariable and multivariavble dataset analysis, resume, is a deep dive on dataset features.</dd>
    <dt>Streamlit Visual App.</dt>
      <dd>After Jupyter Notebook Analysis, i made a simple Streamlit App to CEO of House Rocket, with Maps, Plots and Filters.</dd>
  </dl>
</ul>

<h2>2. Exploratory Data Analysis</h2>
<p>Some insight's to CEO during the EDA and Data Clearing.</p>
<ol>
  <li>Waterfront properties are approximately 30% more expensive than average.</li>
  
![waterfront](https://user-images.githubusercontent.com/75986085/164233298-af60d3c6-6194-42cb-8498-35fb6f042b17.png)
  
  <li>With more than two floors, properties are approximately 5% more expensives than average.</li>
  
  ////
  
  <li>YoY and MoM price of properties is approximately 0%. Houses Pricer do not get more expensiver per year.</li>
  
![month](https://user-images.githubusercontent.com/75986085/164233408-1e212485-d31b-47a0-a9ec-04f34ec19346.png)

![year](https://user-images.githubusercontent.com/75986085/164233446-54be2ba8-23f2-4a3a-a277-08230361beee.png)

</ol>

