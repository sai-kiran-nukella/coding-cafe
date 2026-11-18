"""Native, light Streamlit UI: seller inputs, three numbered sections, optional details."""
import os
import uuid
import json
import hashlib
from pathlib import Path
import pandas as pd
import streamlit as st
import advisor as A
import data_loader as D
import storage as S
import engine as E
from sources import CITIES, SOURCE_URL
from connector import fetch_city
from ui_state import readiness, checks

st.set_page_config(page_title='PriceWise', layout='wide')
money = lambda v: '₹' + format(v, ',.2f')
st.title('PriceWise')
st.write('Check an offer. Compare your options. Keep track of your sales.')
city = st.sidebar.selectbox('City', CITIES, key='city')
frame = S.observations()
catalogue, eligible, other_cities = readiness(frame, city)
series = None
chosen = None
history = None

if 'stock' not in st.session_state:
    st.session_state.stock = 'My stock'
st.sidebar.subheader('Market history')
if len(eligible):
    products = eligible['product'].drop_duplicates().tolist()
    product = st.sidebar.selectbox('Product', products, key='history_product')
    matches = eligible[eligible['product'].eq(product)].reset_index(drop=True)
    labels = [f"{r['market']} · {r['variety']} · {r['grade']} · {r['basis']}" for r in matches.to_dict('records')]
    index = st.sidebar.selectbox('Market and variety', range(len(labels)), format_func=lambda i: labels[i], key='history_choice')
    chosen = matches.iloc[index]
    mask = pd.Series(True, index=frame.index)
    for k in D.KEY:
        mask &= frame[k].eq(chosen[k])
    history = frame.loc[mask].sort_values('date')
    series = history.set_index('date').price
    identity = '|'.join(str(chosen[k]) for k in D.KEY)
    if st.session_state.get('_stock_context') != identity:
        st.session_state.stock = chosen['product']
        st.session_state._stock_context = identity
    st.sidebar.caption(f"{chosen['observations']} reports · last report {chosen['latest']:%d %b %Y}")
else:
    st.sidebar.selectbox('Product', ['Add a qualifying history first'], disabled=True, key='unavailable_product')
    st.sidebar.caption('Calculations work now. Market analysis needs a price history.')

def close_import():
    st.session_state['_show_import'] = False

@st.dialog('Add price history', width='large', on_dismiss=close_import)
def import_dialog():
    st.write(f'Import a genuine CSV or Excel history for {city}, or a file with a city column.')
    uploaded = st.file_uploader('Choose price history', type=['csv', 'xlsx'], key='history_file')
    source = st.text_input('Publisher or source URL', key='source', placeholder='Name the source that published these prices')
    c1,c2 = st.columns(2)
    default_unit = c1.selectbox('Source price unit if absent from file', list(D.FACTORS), key='import_unit', index=1)
    basis = c2.selectbox('Price basis if absent', ['wholesale','retail'], key='basis')
    single = st.checkbox(f'This file is only for {city}; use that city if the column is missing', key='single_city')
    if st.button('Validate and save history', key='import', type='primary'):
        try:
            if uploaded is None:
                raise D.DataError('Choose a price-history file first.')
            f, receipt = D.load_file(uploaded.name, uploaded.getvalue(), source, default_unit, basis, city if single else None)
            S.save_observations(f, receipt)
            saved_catalogue = D.catalogue(S.observations(), city)
            ready_count = int(saved_catalogue.eligible.sum()) if len(saved_catalogue) else 0
            st.session_state['_import_message'] = f"Saved {receipt['accepted']} rows; skipped {receipt['rejected']}. {ready_count} histories ready for {city}."
            close_import()
            st.rerun()
        except (D.DataError, ValueError) as e:
            st.error(str(e))
    with st.expander('File columns and a blank template'):
        st.write('Required: date, product, market, price. Use a city column for multi-city files. Optional: variety, grade, basis and unit.')
        st.write('Dates: YYYY-MM-DD or DD/MM/YYYY. commodity, modal_price and arrival_date column names are also accepted.')
        st.download_button('Download CSV template', b'date,city,product,market,variety,grade,basis,unit,price\n', 'pricewise_template.csv','text/csv')
    with st.expander('Official sources and current-report collection'):
        st.link_button('Open official Indian mandi data', SOURCE_URL)
        st.write('The official API needs DATA_GOV_IN_API_KEY configured locally. A current report adds to stored history; it does not supply a complete historical dataset.')
        if st.button('Fetch official city reports', key='fetch'):
            try:
                f, receipt = fetch_city(os.getenv('DATA_GOV_IN_API_KEY',''),city)
                S.save_observations(f, receipt)
                st.session_state['_import_message'] = f'Saved {len(f)} official rows. Check history readiness below.'
                close_import()
                st.rerun()
            except (D.DataError,ValueError) as e:
                st.error(str(e))

# Keep the next action at the top, rather than in a long technical expander.
status_col, action_col, guide_col = st.columns([3,1,1])
with status_col:
    if series is not None:
        st.success(f'Market history ready · {city} · {chosen["product"]}')
    else:
        st.info(f'{city}: price history needed for market comparison and the model. Seller calculations are ready.')
with action_col:
    if st.button('Add price history', key='add_history_top', width='stretch'):
        st.session_state['_show_import'] = True
with guide_col:
    guide = Path(__file__).parent/'docs'/'PriceWise_Quick_Guide.pdf'
    if guide.exists():
        st.download_button('One-page guide',guide.read_bytes(),'PriceWise_Quick_Guide.pdf','application/pdf',width='stretch')
import_message = st.session_state.pop('_import_message', None)
if import_message:
    st.success(import_message)

st.sidebar.subheader('Your stock and offer')
st.sidebar.caption('Example numbers. Replace them with your actual costs and offers.')
stock = st.sidebar.text_input('Stock name',key='stock')
unit = st.sidebar.selectbox('Quantity unit',['kg','quintal','tonne','litre','dozen'],key='qty_unit')
base = 'kg' if unit in ['kg','quintal','tonne'] else unit
quantity = st.sidebar.number_input('Quantity',min_value=.01,value=500.,step=1.,key='quantity')
cost = st.sidebar.number_input(f'Cost to grow / buy (₹ per {base})',min_value=0.,value=10.,key='cost')
price_unit = st.sidebar.selectbox('Offer price unit',[base]+(['quintal','tonne'] if base=='kg' else []),key='price_unit_'+base)
price = st.sidebar.number_input(f'Your offer (₹ per {price_unit})',min_value=0.,value=15.,key='price_'+price_unit)
transport = st.sidebar.number_input('Transport (₹ total)',min_value=0.,value=300.,key='transport')
with st.sidebar.expander('Fees, stock loss and other costs'):
    other = st.number_input('Other costs (₹ total)',min_value=0.,value=0.,key='other')
    fee = st.number_input('Commission / fees (%)',min_value=0.,max_value=99.,value=0.,key='fee')
    loss = st.number_input('Weight / stock lost (%)',min_value=0.,max_value=100.,value=0.,key='loss')
inputs = dict(quantity=quantity,quantity_unit=unit,cost=cost,price=price,price_unit=price_unit,transport=transport,other=other,fee=fee,loss=loss)
try:
    result = A.calculate(**inputs)
except ValueError as e:
    st.error(f'Check the seller inputs: {e}')
    st.stop()

st.header('1. Check your price and profit')
with st.container(border=True):
    a,b,c = st.columns(3)
    a.metric('Profit at your offer',money(result['profit']))
    b.metric('Break-even price',money(result['break_even'])+f' / {base}' if result['break_even'] is not None else 'No saleable stock')
    c.metric('Saleable stock',f"{result['saleable']:,.2f} {base}")
    if result['profit']>0:
        st.success(f"Your offer leaves {money(result['profit'])} after the costs you entered.")
    elif result['profit']<0:
        st.warning(f"This offer loses {money(-result['profit'])}. Review your selling price and costs.")
    else:
        st.info('This offer breaks even under your entered costs.')
    with st.expander('Show the calculation'):
        st.dataframe(pd.DataFrame({'Item':['Sale revenue','Cost of all starting stock','Fees, transport and other costs','Profit'], 'Amount (₹)':[result['revenue'],result['purchase'],result['expenses'],result['profit']]}),hide_index=True,width='stretch')
    if series is not None:
        if stock.strip().casefold()!=str(chosen['product']).strip().casefold():
            st.info(f"The selected history is for {chosen['product']}; your stock is {stock}. Use matching products before comparing prices.")
        elif chosen['unit']!=base:
            st.info('The history and offer measure different units. Match the unit family before comparing prices.')
        else:
            recent=series.iloc[-30:];median=float(recent.median());lo,hi=recent.quantile([.25,.75]);gap=result['rate']/median-1
            st.subheader('Your offer versus recorded prices')
            p1,p2=st.columns(2)
            p1.metric('Recent median',money(median)+f' / {base}')
            p2.metric('Your offer versus median',f'{gap*100:+.1f}%')
            st.caption(f"{chosen['market']} · {chosen['variety']} · {chosen['basis']} · last report {chosen['latest']:%d %b %Y}. The median is a historical reference, not a guaranteed fair price.")
            st.line_chart(series.rename(f'Reported price (₹ / {base})'))
            with st.expander('Price reference details'):
                st.write(f'The middle half of the latest {len(recent)} reports ranges from {money(lo)} to {money(hi)} per {base}. This describes past prices; it is not a prediction interval.')
            if chosen.age_days>7:
                st.warning('These records are older than seven days. Treat this as a historical comparison.')
    else:
        st.write('Market comparison: waiting for a price history. The profit calculation above is available now.')

st.header('2. Compare your selling options')
waiting_tab,buyer_tab,sales_tab = st.tabs(['Sell now or wait','Compare buyers','Sales book'])
with waiting_tab:
    with st.container(border=True):
        st.write('Change the assumptions below to compare selling today with waiting.')
        c1,c2,c3=st.columns(3)
        days=c1.number_input('Days to wait',min_value=0,max_value=365,value=3,key='days')
        shelf=c2.number_input('Days stock stays saleable',min_value=0,max_value=365,value=7,key='shelf')
        storage=c3.number_input('Storage (₹ / day, whole lot)',min_value=0.,value=50.,key='holding')
        change=st.slider('Assumed selling-price change (%)',-50,100,5,key='change')
        future_loss=st.number_input('Total stock lost if waiting (%)',min_value=0.,max_value=100.,value=5.,key='future_loss')
        cash=st.checkbox('I need payment today',key='cash')
        try:
            wait=A.calculate(**{**inputs,'price':price*(1+change/100),'loss':future_loss,'days':days,'storage':storage})
        except ValueError as e:
            st.error(f'Check the waiting inputs: {e}')
            st.stop()
        n,w=st.columns(2)
        n.metric('Profit if you sell now',money(result['profit']))
        w.metric(f'Profit if you wait {days} days',money(wait['profit']))
        if days>shelf or (cash and days>0):
            st.warning('Waiting does not meet your shelf-life or payment deadline.')
        else:
            delta=wait['profit']-result['profit']
            if delta>0:st.success(f'Waiting adds {money(delta)} under your assumptions.')
            elif delta<0:st.success(f'Selling now keeps {money(-delta)} more under your assumptions.')
            else:st.info('Both options leave the same profit under your assumptions.')
        st.caption('The price change is your assumption, not a forecast. Storage is charged for the whole lot each day.')
        with st.expander('See scenario amounts'):
            st.dataframe(pd.DataFrame([{'Option':'Sell now','Revenue (₹)':result['revenue'],'Profit (₹)':result['profit']},{'Option':f'Wait {days} days','Revenue (₹)':wait['revenue'],'Profit (₹)':wait['profit']}]),hide_index=True)
with buyer_tab:
    with st.container(border=True):
        c1,c2=st.columns(2)
        second_price=c1.number_input(f'Second offer (₹ per {base})',min_value=0.,value=15.,key='second_price')
        second_transport=c2.number_input('Second offer transport (₹ total)',min_value=0.,value=300.,key='second_transport')
        try:
            r2=A.calculate(**{**inputs,'price':second_price,'price_unit':base,'transport':second_transport})
        except ValueError as e:
            st.error(f'Check the second offer: {e}')
            st.stop()
        c1,c2=st.columns(2)
        c1.metric('First offer profit',money(result['profit']))
        c2.metric('Second offer profit',money(r2['profit']))
        difference=r2['profit']-result['profit']
        if difference>0:st.success(f'The second offer leaves {money(difference)} more profit.')
        elif difference<0:st.success(f'The first offer leaves {money(-difference)} more profit.')
        else:st.info('Both offers leave the same profit. Check payment reliability and convenience before choosing.')
        st.caption('Quantity, stock loss, fees and other costs stay the same. Change the second buyer’s price and transport above.')
with sales_tab:
    st.write('Record the first offer only after the sale is complete.')
    with st.form('sale_form',clear_on_submit=False):
        sale_date=st.date_input('Sale date',value=D.today().date(),max_value=D.today().date(),key='sale_date')
        buyer=st.text_input('Buyer / market',key='buyer');note=st.text_input('Note (optional)',key='sale_note')
        done=st.checkbox('This sale is complete; these are the actual amounts',key='complete')
        if st.form_submit_button('Record first offer as a completed sale'):
            if not done or not buyer.strip():st.error('Confirm completion and enter a buyer.')
            else:
                row={'id':str(uuid.uuid4()),'date':str(sale_date),'city':city,'product':stock,'buyer':buyer.strip(),'note':note,'base_unit':base,**result}
                S.add_sale(row);st.success('Sale saved to this local installation.')
    sales=S.sales()
    if sales:
        st.dataframe(pd.DataFrame(sales).drop(columns=['id']),hide_index=True)
        st.metric('Total recorded profit',money(sum(s['profit'] for s in sales)))
        st.download_button('Export sales CSV',A.export_csv(sales),'pricewise_sales.csv','text/csv')
        with st.expander('Delete a sale'):
            delete=st.selectbox('Sale to remove',[r['id'] for r in sales],format_func=lambda i:next(f"{s['date']} | {s['buyer']} | {money(s['profit'])}" for s in sales if s['id']==i))
            confirmation=st.checkbox('Permanently remove the selected sale',key='confirm_delete')
            if st.button('Delete sale'):
                if confirmation:S.delete_sale(delete);st.rerun()
                else:st.error('Confirm deletion first.')
    else:st.write('No completed sales recorded yet.')
    st.download_button('Back up local records (SQL)',S.backup(),'pricewise_backup.sql','text/plain')
report={'city':city,'stock':stock,'inputs':inputs,'result':result,'source':chosen['source'] if chosen is not None else None,'last_observation':str(chosen['latest']) if chosen is not None else None,'waiting_assumptions':{'days':days,'shelf_days':shelf,'price_change_pct':change,'loss_pct':future_loss,'storage_per_day':storage,'payment_today':cash},'waiting_result':wait,'second_offer':r2}
st.download_button('Download calculation report',json.dumps(report,indent=2).encode(),'pricewise_report.json','application/json')

st.header('3. Model and backtest')
with st.container(border=True):
    if series is None:
        st.subheader('Waiting for price history')
        st.write(f'The model has not run for {city}. Add a qualifying history to enable its backtest, training chart and replay.')
        if other_cities:
            st.info('History is saved for '+', '.join(other_cities)+'. Choose the matching city in the sidebar, or add history for '+city+'.')
        if len(catalogue):
            candidates=catalogue.reset_index(drop=True)
            selected_check=st.selectbox('Check an uploaded history',range(len(candidates)),format_func=lambda i:f"{candidates.iloc[i]['product']} · {candidates.iloc[i]['market']} · {candidates.iloc[i]['variety']} · {candidates.iloc[i]['source']}",key='check_history')
            candidate=candidates.iloc[selected_check]
        else:
            candidate=None
        st.dataframe(checks(candidate),hide_index=True,width='stretch')
        st.write('1. Add a genuine CSV or Excel history and its source.\n2. Save it, then select its product and market in the sidebar.\n3. Choose **Run model** here when the history is ready.')
        if st.button('Add history for the model',key='add_history_model',type='primary'):
            st.session_state['_show_import'] = True
        with st.expander('Why a file may not qualify'):
            st.write('Each history is checked separately by city, product, market, variety, grade, unit, retail/wholesale basis and source. Different series are not joined just to reach the minimum count.')
            if len(catalogue):st.dataframe(catalogue[['product','market','observations','span_days','coverage','max_gap_days','reason']],hide_index=True)
    else:
        signature=hashlib.sha256(pd.util.hash_pandas_object(series,index=True).values.tobytes()+identity.encode()).hexdigest()
        st.success(f'History ready: {len(series)} reports for {chosen["product"]} in {city}.')
        st.caption('The model predicts the next reported observation. It does not promise a price for a specific sale date.')
        if st.button('Run model',key='run_model',type='primary'):
            st.session_state.pop('_analysis',None)
            st.session_state.pop('_analysis_signature',None)
            try:
                with st.status('Training and checking unseen prices…',expanded=False) as status:
                    a=E.analyze(series)
                    st.session_state['_analysis']=a
                    st.session_state['_analysis_signature']=signature
                    status.update(label='Analysis ready: backtest, training and replay completed.',state='complete')
            except (ValueError,FloatingPointError,ArithmeticError) as e:
                st.error(f'The analysis could not finish: {e}. Review the selected history and try again.')
        if st.session_state.get('_analysis_signature')==signature:
            a=st.session_state['_analysis']
            backtest,training,replay=st.tabs(['Backtest','Training','Historical replay'])
            with backtest:
                st.write(f"Selected using validation data: **{a['chosen']}**.")
                m1,m2=st.columns(2)
                m1.metric('Selected-method error',money(a['scores'][a['chosen']])+f" / {chosen['unit']}")
                m2.metric('Last-price baseline error',money(a['scores']['Last price'])+f" / {chosen['unit']}")
                if a['has_skill'] and chosen.age_days<=7:
                    st.success(f"Next reported-price estimate: {money(a['forecast'])} per {chosen['unit']}.")
                elif chosen.age_days>7:
                    st.info('Backtest complete. A current forecast is withheld because the latest report is older than seven days.')
                else:
                    st.info('Backtest complete. The selected method did not improve enough on the last-price baseline, so no improved forecast is issued.')
                st.line_chart(a['test'][['Observed',a['chosen']]].loc[:,lambda x:~x.columns.duplicated()])
                with st.expander('All model errors and test details'):
                    st.dataframe(pd.DataFrame({'Method':list(a['scores']),'Mean absolute error (₹)':list(a['scores'].values())}),hide_index=True)
                    st.write(f"Final holdout improvement: {100*a['skill']:.1f}%. Training ended {a['train_end']:%d %b %Y}; validation ended {a['validation_end']:%d %b %Y}.")
                    st.caption('First 60% training, next 20% validation, final 20% testing. Scaling fits training only. Published-vintage data is used; release delays are not simulated.')
                st.download_button('Export backtest',a['test'].to_csv().encode(),'pricewise_backtest.csv','text/csv')
            with training:
                st.write(f"NumPy network trained. Best validation epoch: **{a['best_epoch']}**.")
                st.line_chart(pd.Series(a['losses'],name='Training loss'))
                st.caption('A falling training loss does not establish forecasting skill. Check the unseen backtest too.')
            with replay:
                step=st.slider('Reveal unseen reports',1,len(a['test']),1,key='replay_step_'+signature[:10])
                st.line_chart(a['test'].iloc[:step][['Observed',a['chosen']]].loc[:,lambda x:~x.columns.duplicated()])
                st.dataframe(a['test'].iloc[:step])
        else:
            st.info('Ready to train. Choose Run model to generate the backtest, training chart and historical replay for this selection.')
    if series is not None:
        with st.expander('Selected history and source'):
            st.write(f"Source declared: {chosen['source']}. Latest observation: {chosen['latest']:%d %b %Y}.")
            st.dataframe(checks(chosen),hide_index=True)
            st.download_button('Export selected price history',A.export_csv(history.assign(date=history.date.dt.strftime('%Y-%m-%d')).to_dict('records')),'pricewise_history.csv','text/csv')

if st.session_state.get('_show_import', False):
    import_dialog()
