def diff_to_days(x):
    return x / np.timedelta64(1, 's') / 60 / 60 / 24

def _get_features(data, t_start, t_end):
    data_no_duplicates = data.drop_duplicates("chq_id")
    data_negative      = data[data['sales_sum']<0]
    
    t_list = list(sorted(data_no_duplicates['chq_date'].unique()))
    deltas = diff_to_days(np.diff(t_list)) #интервалы между датами транзакций
    bins = [0, 1, 7, 14, 21, 28, 56, 112, 224, 448]
    res = {
           't_min'  : data['chq_date'].min(),          # Даты (первая, последняя и весь список)
           't_max'  : data['chq_date'].max(),  
           '_t_list' : t_list,
        
           'items_N'               : data['chq_id'].nunique(),              # Количество пробитых товаров
           'items_sales_sum_sum'   : data['sales_sum'].sum(),     # Сколько потратил
           'items_sales_count_sum' : data['sales_count'].sum(), # Сколько покупок сделал

           'items_N_negative' : data_negative['chq_id'].nunique(),              # Количество пробитых товаров
           'items_sales_sum_sum_negative':data_negative['sales_sum'].sum(),     # Сколько потратил
           'items_sales_count_sum_negative':data_negative['sales_count'].sum(), # Сколько покупок сделал
           
           "item_price_mean" : data['chq_id_sum_sales_count'].mean(), # Средняя цена товара
           "item_price_max"  : data['chq_id_len'].max(),              # Максимальная цена товара
           "item_price_min"  : data['chq_id_sum_sales_sum'].min(),    # Минимальная цена товара
           
        
           'chq_id_sum_sales_count_mean':data_no_duplicates['chq_id_sum_sales_count'].mean(),   # Средний количество покупок
           'chq_id_len_mean':data_no_duplicates['chq_id_len'].mean(),                           # Средее количнтво товаров
           'chq_id_sum_sales_sum_mean':data_no_duplicates['chq_id_sum_sales_sum'].mean(),       # Средняя цена товаров
                   
           'chq_id_sum_sales_count_min':data_no_duplicates['chq_id_sum_sales_count'].min(),  # То-же самое для минимума
           'chq_id_len_min':data_no_duplicates['chq_id_len'].min(),
           'chq_id_sum_sales_sum_min':data_no_duplicates['chq_id_sum_sales_sum'].min(),
            
           
           'chq_id_sum_sales_count_max':data_no_duplicates['chq_id_sum_sales_count'].max(), # То-же самое для максимума
           'chq_id_len_max':data_no_duplicates['chq_id_len'].max(),
           'chq_id_sum_sales_sum_max':data_no_duplicates['chq_id_sum_sales_sum'].max(),
        
           'promo_rate_count': (data['is_promo']==1).mean(),    # Доля промо
           'promo_rate_sales_sum': (data.loc[(data['is_promo']==1), 'sales_sum'].sum()+1)/(data['sales_sum'].sum()+1),
           'promo_rate_sales_count': (data.loc[(data['is_promo']==1), 'sales_count'].sum()+1)/(data['sales_count'].sum()+1),
        
            "delta_median" : np.median(deltas) if len(deltas)>0 else np.nan, # 
            "delta_min"    : np.min(deltas) if len(deltas)>0 else np.nan,
            "delta_max"    : np.max(deltas) if len(deltas)>0 else np.nan,
            "delta_std"    : np.std(deltas) if len(deltas)>0 else np.nan,
            "delta_moda"   : np.argmax(np.bincount(deltas.astype(int))) if len(deltas)>0 else np.nan,
          }
    
    # Признаки сколько активности пользователя по дням
    res['days_from_start'] = (res['t_min']-t_start).days
    res['days_until_end']  = (t_end-res['t_max']).days
    res['days_max']        = (res['t_max']-res['t_min']).days
    
    # Последний поход в магазин
    res['deltas_-1'] = deltas[-1] if len(deltas)>0 else np.nan
    res['deltas_-2'] = deltas[-2] if len(deltas)>1 else np.nan
    res['deltas_-1_frac_-2'] = deltas[-2]/deltas[-1]if len(deltas)>1 else np.nan
    
    # Признаки про магазин
    for key in ['plant__is_SM', 'plant__is_city_St. Petersburg', 'plant__is_city_Other', 'plant__is_city_Moscow']:
        res[key+"_mean"] = data_no_duplicates[key].mean()
                
    # Признаки про товар
    for key in ['chq_id_material__is_private_label', 'chq_id_material__is_alco', 'chq_id_material__is_food']:
        res[key+"_mean"] = data_no_duplicates[key].mean()
                
    # Средняя цена покупки
    res['chq_id__average_transaction'] = res['chq_id_sum_sales_sum_mean'] / res['chq_id_sum_sales_count_mean']

    # Позитивные транзакции
    res['items_N_positive']               = res['items_N']-res['items_N_negative']
    res['items_sales_sum_sum_positive']   = res['items_sales_sum_sum']-res['items_sales_sum_sum_negative']
    res['items_sales_count_sum_positive'] = res['items_sales_count_sum']-res['items_sales_count_sum_negative']
    
    
    # Отношение негативного к позитивному
    ##res['negative_pos_rate_len_positive'] = res['negative_pos_rate_len']  -res['negative_pos_rate_len_negative']  
    for l in range(1, len(bins)):
        res[f'delta_bin_{bins[l-1]}_{bins[l],}'] = np.mean([((bins[l-1]<=x) and (x<bins[l]))  for x in deltas])
    
    
    for c in range(2):
        res[f'plant_key_{c}'] = np.nan
        res[f'plant_val_{c}'] = np.nan 
        res[f'material_key_{c}'] = np.nan
        res[f'material_val_{c}'] = np.nan 
        
    res['plant_N']    = data['plant'].nunique()
    res['material_N'] = data_no_duplicates['material'].nunique()
    
    # Самые популярные города по количеству и из доля
    for c, (key, val) in enumerate(data_no_duplicates['plant'].value_counts(True).to_dict().items()):
        if c>=2:
            break

        res[f'plant_key_{c}'] = key
        res[f'plant_val_{c}'] = val     
    #два топовых магазина для юзера
        
    # Самые популярные товары по количеству и из доля    
    for c, (key, val) in enumerate(data['material'].value_counts(True).to_dict().items()):
        if c>=2:
            break
            
        res[f'material_key_{c}'] = key
        res[f'material_val_{c}'] = val  
    
    # Добавить join на metrials из таблицы 'materials.txt'
    # Добавить join на plant из таблицы пользователей если для каждого магазина взять максимальное количество пользователей с городом, это должно помоч

    return pd.Series(res)  
