function split_attributes(df)
    num_df = hcat( select(df, findall(col -> eltype(col) <: Int64, eachcol(df))), 
                    select(df, findall(col -> eltype(col) <: Float64, eachcol(df))) )
    
    no_num_df = df[:, [k for k in names(df) if k ∉ names(num_df)]]
    
    return Dict("Num Df" => num_df, "No-Num Df" => num_df)
end;

function plot_hist_info(df, x)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[1].hist(df[:, x], label=x,  color="r", histtype="step", bins=100)
    ss.probplot(df[:, x], plot=ax[2])
    ax[2].text(.1, .5, x, transform=ax[2].transAxes)
end;

function plot_year_price(aux1, aux2)
    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    ax[1].bar(aux1.month, aux1.price_median, color="r")
    ax[2].bar(aux1.year, aux1.price_median, color="k")
    ax[3].scatter(aux2.yr_built, aux2.price_median, color="k")
    ax[4].hist(aux2.price_median, color="r", bins=50);
    plt.tight_layout(h_pad=2)
    for k in zip(["Price per Month", "Price per Year", "Price Per Yr Built", "Year Built Price Dist"], [1, 2, 3, 4])
        ax[k[2]].set_title(k[1])
    end;
end;

function plot_price_month(aux1, aux2)
    plt.subplots(figsize=(15, 10))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    ax1.bar(aux1.month_name, aux1.price_median, color="k")
    ax1.set_xticklabels(df4.month_name, rotation=45)
    ax2.bar(aux2.day_name, aux2.price_median, color="k")
    ax3.scatter(df4.month, df4.price, color="r");
end;

function plot_price_floors(aux1, aux12, aux2)
    plt.subplots(figsize=(15, 10))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    ax1.bar(aux1.floors, aux1.price_median, .35, color="k")
    ax2.bar(aux12.floors, aux12.floors_length, .35, color="r")
    ax3.scatter(aux2.floors, aux2.price_median)

    ax1.set_title("Median price per Floor")
    ax2.set_title("Total of Floors");
    ax3.set_title("Scatter per Price/Floor")
end;

function plot_price_waterfront(aux1, aux2, aux3)
    plt.subplots(figsize=(15, 10))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    ax1.bar(aux1.waterfront, aux1.price_median, color="k")
    ax2.hist(df4[df4.waterfront .== 1, "price"], bins=100, color="r", label="With Waterfront")
    ax2.hist(sort(df4[df4.waterfront .== 0, "price"])[21250:end], bins=100, color="b", label="Without Waterfront")
    ax3.scatter(aux2.month_name, aux2.price_median, label="With Waterfront", color="r")
    ax3.scatter(aux3.month_name, aux3.price_median, label="Without Waterfront", color="k")
    ax1.set_title("Median Price Diference")
    ax2.set_title("Most Expensive Houses");
    ax3.set_title("Median Price per Month");
    ax2.legend();
    ax3.legend();
end;

function plot_pie_condition(aux0, aux2)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[1].pie(aux2.price_median, colors=["r", "k", "navy", "b", "gold"], labels=aux2.condition, 
              explode=(0, .0, .0, .0, .11), startangle=180, autopct="%1.1f%%")
    ax[1].set_title("Price per Condition")
    ax[2].bar(aux1.condition, aux1.price_mean, color="k")
    ax[2].set_title("Price per Condition");
end;

function plot_3d_day_fig(aux)
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(aux.day, aux.floors, aux.price_median)
    ax.set_title("3D Day/Floor/Price")
    ax.set_xlabel("Days", color="b")
    ax.set_ylabel("Floors")
    ax.set_zlabel("Price", color="r")
end;

function plot_3d_condition_fig(aux)
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(aux.yr_built, aux.condition, aux.price_median)
    ax.set_title("3D Day/Floor/Price")
    ax.set_xlabel("Days", color="b")
    ax.set_ylabel("Condition")
    ax.set_zlabel("Price", color="r")
end;