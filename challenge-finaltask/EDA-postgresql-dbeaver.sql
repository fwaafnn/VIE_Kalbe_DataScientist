-- Afwa Afini_Data Scientist_VIE Kalbe

-- query 1: rata-rata umur customer jika dilihat dari marital statusnya 
select marital_status, round(avg(age)) as age_average from customer c group by marital_status

-- query 2: rata-rata umur customer jika dilihat dari gender nya
select gender, round(avg(age)) as age_average from customer c group by gender 

-- query 3: nama store dengan total quantity terbanyak
select t.storeid, s.store_name, sum(t.qty) as total_qty
from transaction t
inner join store s 
on t.storeid = s.store_id
group by t.storeid, s.store_name  
order by sum(t.qty) desc
limit 1

-- query 4: nama produk terlaris dengan total amount terbanyak
select t.productid, p.product_name, sum(t.totalamount) as total_amount
from transaction t
inner join product p 
on t.productid = p.product_id
group by t.productid, p.product_name
order by sum(t.totalamount) desc
limit 1


