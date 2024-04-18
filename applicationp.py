from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import category_encoders as ce
import matplotlib.pyplot as plt
import mpld3

app = Flask(__name__)

# Load CSV files
cost_sell_df = pd.read_csv('cost_sell_df.csv')
filtered_df = pd.read_csv('filtered_df.csv')
combined_df = pd.read_csv('combined_df.csv')

# Load models
'''
with open('onehotmodel/RandomForestCSfinal.pkl', 'rb') as model_file1:
    model1 = pickle.load(model_file1)

with open('onehotmodel/ExtraTreesQtyf.pkl', 'rb') as model_file2:
    model2 = pickle.load(model_file2)

with open('onehotmodel/RandomForestQtyf.pkl', 'rb') as model_file3:
    model3 = pickle.load(model_file3)

'''
model1 = pickle.load(open('RandomForestCSfinal.pkl','rb'))
model2 = pickle.load(open('ExtraTreesQtyf.pkl','rb'))
model3 = pickle.load(open('ExtraTreesQtyC.pkl','rb'))


'''
model1 = joblib.load('RandomForestCPtoSP.joblib')
model2 = joblib.load('ExtraTreesQtyf.joblib')
model3 = joblib.load('ExtraTreesQtyC.joblib')
'''

# Get unique product names
top_products = (filtered_df['ProductName'].value_counts().nlargest(25).index
                .union(combined_df['ProductName'].value_counts().nlargest(25).index)).tolist()

@app.route('/')
def index():
    return render_template('index.html', products=top_products)

@app.route('/get_cost_range', methods=['POST'])
def get_cost_range():
    p_name = request.json['product']
    cost_range = {
        'min_cost': cost_sell_df[cost_sell_df['ProductName'] == p_name]['CostPrice'].min(),
        'max_cost': cost_sell_df[cost_sell_df['ProductName'] == p_name]['CostPrice'].max()
    }
    return jsonify(cost_range)

'''@app.route('/calculate', methods=['POST'])
def calculate():
    p_name = request.form.get('product')
    cost_price = float(request.form.get('cost-price'))
    duration = int(request.form.get('duration'))  '''


@app.route('/calculate', methods=['POST'])
def calculate():
    p_name = request.form.get('product')
    cost_price = float(request.form.get('cost-price'))
    duration = int(request.form.get('duration'))

    # Step 6: Use model1 to predict RetailSPrice for the selected product
    test_df = pd.DataFrame({'ProductName': [p_name], 'CostPrice': [cost_price]})
    convert_dict = {'ProductName': str, 'CostPrice': float}
 
    test_df = test_df.astype(convert_dict)
    #start_price = model1.predict(test_df.astype('float64'))[0]
    
    start_price = model1.predict(test_df)[0]
    print(start_price)
    # Step 8: Calculate end_price
    end_price = cost_price + 0.20*cost_price

    # Step 9: Generate a numpy.arange from start_price to end_price with a gap of 2
    price_list = np.arange(start_price, end_price + 2, 2) #start_price
    price_list_series = pd.Series(price_list)

    # Step 11: Make new predict2_df
    predict2_df = pd.DataFrame({
        'ProductName': [p_name] * len(price_list_series),
        'RetailSPrice': price_list_series,
        'DaysUsed': [duration] * len(price_list_series)
    })
    convert_dict1 = {'ProductName': str, 'RetailSPrice': float, 'DaysUsed': int}
    predict2_df = predict2_df.astype(convert_dict1)
    test_df = test_df.astype(convert_dict)
    # Step 12: Make a function that passes the model and predict2_df and returns predicted outcomes
    def predict_outcomes(model, df):
        predicted_values = pd.Series(model.predict(df))  #df.astype('float64')
        return predicted_values

    # Step 13: Determine which model to use based on ProductName
    if p_name in filtered_df['ProductName'].values:
        model_to_use = model2
    elif p_name in combined_df['ProductName'].values:
        model_to_use = model3
    else:
        return "Invalid product selected."

    # Step 14: Make new column 'TotalQty' for predict2_df
    predict2_df['TotalQty'] = predict_outcomes(model_to_use, predict2_df)
    predict2_df['TotalQty'] = predict2_df['TotalQty'].round()
    print(predict2_df[['ProductName','RetailSPrice','TotalQty','DaysUsed']])
    # Step 15: Make new column 'Profit' for predict2_df
    predict2_df['RetailSPrice'] = predict2_df['RetailSPrice'].round()
    predict2_df['Profit'] = (predict2_df['RetailSPrice'] - cost_price) * predict2_df['TotalQty']

    # Step 16: Remove rows with negative values in the 'Profit' column
    predict2_df = predict2_df[predict2_df['Profit'] >= 0]
    predict2_df = pd.DataFrame(predict2_df)
    # Step 17: Find the row with the maximum profit
    optimal_row = predict2_df.loc[predict2_df['Profit'].idxmax()]

    # Step 18: Find the row with the maximum TotalQty
    max_quantity_row = predict2_df.loc[predict2_df['TotalQty'].idxmax()]
    # Filter the DataFrame to include only rows with the maximum quantity
    #max_quantity_rows = predict2_df[predict2_df['TotalQty'] == predict2_df.loc[max_quantity_row, 'TotalQty']]

    # Among the rows with maximum quantity, find the row with maximum price
    #max_pricequant_row = max_quantity_rows.loc[max_quantity_rows['RetailSPrice'].idxmax()]

    max_quants = predict2_df[predict2_df['TotalQty']==predict2_df['TotalQty'].max()]
    max_pricequant_row = max_quants.loc[max_quants['RetailSPrice'].idxmax()]
    # Step 19: Generate a scatter plot
    fig_PriceVsQuantity = px.scatter(cost_sell_df[cost_sell_df['ProductName'] == p_name],
                                     x="RetailSPrice", y="SQty",trendline="ols") #
    # Add the trendline
    #fig_PriceVsQuantity = fig_PriceVsQuantity.add_trendline("ols")
    fig_PriceVsQuantity.update_layout(
        height=600,
        width=800,
        title="Scatter Plot of Retail Price vs. Quantity Sold for " + p_name
    )
    
    #scatter_data = pio.to_json(fig_PriceVsQuantity, full_figure=True)
    
    #scatter_plot_html = f'''
       # <div>
        #    <script>
         #       var scatter_data = {scatter_data};
          #      Plotly.newPlot('scatter-plot', scatter_data.data, scatter_data.layout);
           # </script>
        #</div>
       # '''   
    scatter_plot_html = pio.to_html(fig_PriceVsQuantity, full_html=False, include_plotlyjs=True)

    scatter_plot_html = f'''
    <div>
        {scatter_plot_html}
    </div>
    '''
    # Start of trend plot
    # Create the main plot with price on the x-axis and profit on the left y-axis
    fig = go.Figure()

    # Scatter plot for Profit
    fig.add_trace(go.Scatter(x=predict2_df['RetailSPrice'], y=predict2_df['Profit'],
                         mode='markers+lines', name='Profit', line=dict(color='blue')))

    # Set the layout for the main y-axis
    fig.update_layout(
        xaxis=dict(title='Price'),
        yaxis=dict(title='Profit', side='left', color='blue')
    )

    # Create a secondary y-axis for quantity
    fig.add_trace(go.Scatter(x=predict2_df['RetailSPrice'], y=predict2_df['TotalQty'],
                             mode='markers+lines', name='Quantity', line=dict(color='green')))

    # Set the layout for the secondary y-axis
    fig.update_layout(
        yaxis2=dict(title='Quantity', overlaying='y', side='right', color='green')
    )

    # Set the title
    fig.update_layout(title='Price, Profit, and Quantity')

    # Customize the layout as needed
    fig.update_layout(height=600, width=800)

    # Save the HTML file
    trend_plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=True)

    # Embed the HTML code into a string
    trend_plot_html = f'''
    <div>
    {trend_plot_html}
    </div>
    '''
    #end of trend plot
    # Now 'scatter_plot_html' contains the HTML code for the Plotly plot
    fig, ax1 = plt.subplots()
    ax1.plot(predict2_df['RetailSPrice'], predict2_df['Profit'], color='red', linestyle='-', label='Revenue')
    ax1.set_xlabel('Price')
    ax1.set_ylabel('Profit', color='blue')
    plt.legend()
    ax2 = ax1.twinx()
    ax2.plot(predict2_df['RetailSPrice'], predict2_df['TotalQty'], color='green', linestyle='-', label='Quantity')
    ax2.set_ylabel('Quantity', color='green')

    plt.title('Price, Profit, and Quantity')
    plt.legend() #added
    plt.show() #added
    # Convert the Matplotlib plot to HTML using mpld3
    plot_html = mpld3.fig_to_html(fig)

    # Close the Matplotlib plot to avoid memory leaks
    plt.close()
    return jsonify({
        'optimal_product_name': optimal_row['ProductName'],
        'optimal_price': optimal_row['RetailSPrice'],
        'optimal_quantity': optimal_row['TotalQty'],
        'optimal_profit': optimal_row['Profit'],
        'more_quantity_product_name': max_pricequant_row['ProductName'],
        'more_quantity_price': max_pricequant_row['RetailSPrice'],
        'more_quantity': max_pricequant_row['TotalQty'],
        'more_quantity_profit': max_pricequant_row['Profit'],
        'scatter_plot': scatter_plot_html,
       # 'trend_plot': trend_plot_html,
        'plot_html': plot_html
    })
    # Step 20: Return the results to the result.html template
    #return render_template('index.html', optimal_row=optimal_row, more_quantity_row=more_quantity_row, plot=fig_PriceVsQuantity.to_html())
    #return render_template('result.html', result=result, more_quantity=more_quantity)


if __name__ == '__main__':
    app.run(debug=True)
