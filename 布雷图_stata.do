use "D:\github\research\Breteau index\布雷图.dta",clear
drop date survey_site survey_site survey_section longitude latitude id place
gen spring=1 if month>=3&month<6
gen summer=1 if month>=6&month<9
gen autumn=1 if month>=9&month<12
gen winter=1 if month==12|month<2
replace spring=0 if spring!=1
replace summer=0 if summer!=1
replace autumn=0 if autumn!=1
replace winter=0 if winter!=1
drop households_sum bonsai_pos tank_pos containers_pos channels_pos hole_pos tires_pos litter_pos basement_pos other_pos albopictus aegypti sum_pos households_pos sum_standing pos_standing month

*全变量自相关
local v "weather door bonsai tank containers channels hole tires litter basement other breteau_index households_ins tem_low tem_low7 tem_low6 tem_low5 tem_low4 tem_low3 tem_low2 tem_low1 tem_mean tem_mean7 tem_mean6 tem_mean5 tem_mean4 tem_mean3 tem_mean2 tem_mean1 tem_high tem_high7 tem_high6 tem_high5 tem_high4 tem_high3 tem_high2 tem_high1 sunshine_hours sunshine_hours7 sunshine_hours6 sunshine_hours5 sunshine_hours4 sunshine_hours3 sunshine_hours2 sunshine_hours1 precipitation precipitation7 precipitation6 precipitation5 precipitation4 precipitation3 precipitation2 precipitation1 humidity humidity7 humidity6 humidity5 humidity4 humidity3 humidity2 humidity1 pressure pressure7 pressure6 pressure5 pressure4 pressure3 pressure2 pressure1 wind wind7 wind6 wind5 wind4 wind3 wind2 wind1"
logout, save(all_corr) excel replace: pwcorr_a `v'

*机器学习自相关 RF
local v "containers tank bonsai pressure7 precipitation7 other wind3 wind5 tires wind sunshine_hours4 wind1 wind6 wind7 humidity7 sunshine_hours precipitation3 sunshine_hours2"
logout, save(RF_corr) excel replace: pwcorr_a `v'

*机器学习自相关 DT
local v "tank containers bonsai tires tem_mean3 pressure3 precipitation7 humidity1 precipitation3 tem_high3 precipitation1 other wind2 humidity5 tem_high2 pressure4 tem_low7 sunshine_hours tem_high7 humidity"
logout, save(DT_corr) excel replace: pwcorr_a `v'

*机器学习自相关 ALL_ML
local v "containers tank bonsai pressure7 precipitation7 other wind3 wind5 tires wind sunshine_hours4 wind1 wind6 wind7 humidity7 sunshine_hours precipitation3 sunshine_hours2 tem_mean3 pressure3 humidity1 tem_high3 precipitation1 wind2 humidity5 tem_high2 pressure4 tem_low7 tem_high7 humidity"
logout, save(ML_corr) excel replace: pwcorr_a `v'

***滞后效应模型
*平均
gen breteauround=round(breteau)
reg breteau_index door bonsai tank containers channels hole tires litter basement other tem_low tem_mean tem_high sunshine_hours precipitation humidity pressure wind spring summer autumn winter
eststo tab1
stepwise, pr(.2):reg breteau_index door bonsai tank containers channels hole tires litter basement other tem_low tem_mean tem_high sunshine_hours precipitation humidity pressure wind spring summer autumn winter
eststo tab2
gam breteauround door bonsai tank containers channels hole tires litter basement other tem_low tem_mean tem_high sunshine_hours precipitation humidity pressure wind spring summer autumn winter, family(binomial) link(logit)
eststo tab3
esttab using breteau_habitat_weather_mean.csv,star noomit nobasel r2 aic bic numbers replace
*滞后
forvalues i= 1(1)7 {
	reg breteau_index door bonsai tank containers channels hole tires litter basement other tem_low`i' tem_mean`i' tem_high`i' sunshine_hours`i' precipitation`i' humidity`i' pressure`i' wind`i' spring summer autumn winter
	eststo tab1
	stepwise, pr(.2):reg breteau_index door bonsai tank containers channels hole tires litter basement other tem_low`i' tem_mean`i' tem_high`i' sunshine_hours`i' precipitation`i' humidity`i' pressure`i' wind`i' spring summer autumn winter
	eststo tab2
	gam breteauround door bonsai tank containers channels hole tires litter basement other tem_low`i' tem_mean`i' tem_high`i' sunshine_hours`i' precipitation`i' humidity`i' pressure`i' wind`i' spring summer autumn winter, family(binomial) link(logit)
	eststo tab3
	esttab using breteau_habitat_weather_lag`i'.csv,star noomit nobasel r2 aic bic numbers replace

}


*****机器学习扩大筛选范围 此段代码废弃
forvalues i= 1(1)7 {
	label variable tem_low`i' "`i'日前日最低气温(℃)"
	label variable tem_mean`i' "`i'日前平均气温(℃)"
	label variable tem_high`i' "`i'日前日最高气温(℃)"
	label variable sun_dur`i' "`i'日前日照时数(h)"
	label variable precipitation`i' "`i'日前降水量(mm)"
	label variable hum_mean`i' "`i'日前平均相对湿度(%)"
	label variable pre_mean`i' "`i'日前平均气压(hpa)"
	label variable wind_mean`i' "`i'日前平均风速(m/s)"

}
label variable breteau "布雷图指数"
label variable tem_low "日最低气温(℃)"
label variable tem_mean "平均气温(℃)"
label variable tem_high "日最高气温(℃)"
label variable sun_dur "日照时数(h)"
label variable precipitation "降水量(mm)"
label variable hum_mean "平均相对湿度(%)"
label variable pre_mean "平均气压(hpa)"
label variable wind_mean "平均风速(m/s)"
keep hum_mean hum_mean6 tem_low7 tem_low4 tem_high2 pre_mean4 hum_mean5 wind_mean4 hum_mean4 pre_mean2 wind_mean2 pre_mean3 wind_mean hum_mean1 sun_dur2 wind_mean7 wind_mean6 hum_mean2 precipitation4 pre_mean6 hum_mean3 wind_mean3 wind_mean5 pre_mean5 pre_mean7 hum_mean7 precipitation7 tem_high6 tem_low wind_mean4 pre_mean3 tem_high5 pre_mean7 wind_mean7 sun_dur pre_mean2 sun_dur2 precipitation7 hum_mean2 tem_low7 wind_mean5 pre_mean5 wind_mean3 breteau



reg breteau tem_low tem_low7 tem_low4 tem_high6 tem_high5 tem_high2 sun_dur sun_dur2 precipitation7 precipitation4 hum_mean hum_mean7 hum_mean6 hum_mean5 hum_mean4 hum_mean3 hum_mean2 hum_mean1 pre_mean7 pre_mean6 pre_mean5 pre_mean4 pre_mean3 pre_mean2 wind_mean wind_mean7 wind_mean6 wind_mean5 wind_mean4 wind_mean3 wind_mean2
eststo tab1
stepwise, pr(.2):reg breteau tem_low tem_low7 tem_low4 tem_high6 tem_high5 tem_high2 sun_dur sun_dur2 precipitation7 precipitation4 hum_mean hum_mean7 hum_mean6 hum_mean5 hum_mean4 hum_mean3 hum_mean2 hum_mean1 pre_mean7 pre_mean6 pre_mean5 pre_mean4 pre_mean3 pre_mean2 wind_mean wind_mean7 wind_mean6 wind_mean5 wind_mean4 wind_mean3 wind_mean2
eststo tab2
gam breteauround tem_low tem_low7 tem_low4 tem_high6 tem_high5 tem_high2 sun_dur sun_dur2 precipitation7 precipitation4 hum_mean hum_mean7 hum_mean6 hum_mean5 hum_mean4 hum_mean3 hum_mean2 hum_mean1 pre_mean7 pre_mean6 pre_mean5 pre_mean4 pre_mean3 pre_mean2 wind_mean wind_mean7 wind_mean6 wind_mean5 wind_mean4 wind_mean3 wind_mean2, family(binomial) link(logit)
eststo tab3
gam breteauround sun_dur sun_dur2 precipitation7 precipitation4 hum_mean hum_mean7 hum_mean6 hum_mean5 hum_mean4 hum_mean3 hum_mean2 hum_mean1 pre_mean7 pre_mean6 pre_mean5 pre_mean4 pre_mean3 pre_mean2 wind_mean wind_mean7 wind_mean6 wind_mean5 wind_mean4 wind_mean3 wind_mean2, family(binomial) link(logit)
eststo tab4
reg breteau sun_dur sun_dur2 precipitation7 precipitation4 hum_mean hum_mean7 hum_mean6 hum_mean5 hum_mean4 hum_mean3 hum_mean2 hum_mean1 pre_mean7 pre_mean6 pre_mean5 pre_mean4 pre_mean3 pre_mean2 wind_mean wind_mean7 wind_mean6 wind_mean5 wind_mean4 wind_mean3 wind_mean2
eststo tab5
stepwise, pr(.2):reg breteau sun_dur sun_dur2 precipitation7 precipitation4 hum_mean hum_mean7 hum_mean6 hum_mean5 hum_mean4 hum_mean3 hum_mean2 hum_mean1 pre_mean7 pre_mean6 pre_mean5 pre_mean4 pre_mean3 pre_mean2 wind_mean wind_mean7 wind_mean6 wind_mean5 wind_mean4 wind_mean3 wind_mean2
eststo tab6

gen breteauround=round(breteau)
**DT
eststo clear
reg breteau_index tank containers bonsai tires tem_mean3 pressure3 precipitation7 humidity1 precipitation3 tem_high3 precipitation1 other wind2 humidity5 tem_high2 pressure4 tem_low7 sunshine_hours tem_high7 humidity
eststo tab1
stepwise, pr(.2):reg breteau_index   tank containers bonsai tires tem_mean3 pressure3 precipitation7 humidity1 precipitation3 tem_high3 precipitation1 other wind2 humidity5 tem_high2 pressure4 tem_low7 sunshine_hours tem_high7 humidity
eststo tab2
gam breteauround tank containers bonsai tires tem_mean3 pressure3 precipitation7 humidity1 precipitation3 tem_high3 precipitation1 other wind2 humidity5 tem_high2 pressure4 tem_low7 sunshine_hours tem_high7 humidity , family(binomial) link(logit)
eststo tab3
esttab using breteau_DT.csv,l star noomit nobasel r2 aic bic numbers replace

**RF
eststo clear
reg breteau_index  containers tank bonsai pressure7 precipitation7 other wind3 wind5 tires wind sunshine_hours4 wind1 wind6 wind7 humidity7 sunshine_hours precipitation3 sunshine_hours2            
eststo tab1
stepwise, pr(.2):reg breteau_index  containers tank bonsai pressure7 precipitation7 other wind3 wind5 tires wind sunshine_hours4 wind1 wind6 wind7 humidity7 sunshine_hours precipitation3 sunshine_hours2            
eststo tab2
gam breteauround containers tank bonsai pressure7 precipitation7 other wind3 wind5 tires wind sunshine_hours4 wind1 wind6 wind7 humidity7 sunshine_hours precipitation3 sunshine_hours2  , family(binomial) link(logit)
eststo tab3
esttab using breteau_RF.csv,l star noomit nobasel r2 aic bic numbers replace

**合并
eststo clear
reg breteau_index containers tank bonsai pressure7 precipitation7 other wind3 wind5 tires wind sunshine_hours4 wind1 wind6 wind7 humidity7 sunshine_hours precipitation3 sunshine_hours2 tem_mean3 pressure3 humidity1 tem_high3 precipitation1 wind2 humidity5 tem_high2 pressure4 tem_low7 tem_high7 humidity
eststo tab1
stepwise, pr(.2):reg breteau_index containers tank bonsai pressure7 precipitation7 other wind3 wind5 tires wind sunshine_hours4 wind1 wind6 wind7 humidity7 sunshine_hours precipitation3 sunshine_hours2 tem_mean3 pressure3 humidity1 tem_high3 precipitation1 wind2 humidity5 tem_high2 pressure4 tem_low7 tem_high7 humidity
eststo tab2
gam breteauround containers tank bonsai pressure7 precipitation7 other wind3 wind5 tires wind sunshine_hours4 wind1 wind6 wind7 humidity7 sunshine_hours precipitation3 sunshine_hours2 tem_mean3 pressure3 humidity1 tem_high3 precipitation1 wind2 humidity5 tem_high2 pressure4 tem_low7 tem_high7 humidity, family(binomial) link(logit)
eststo tab3
esttab using breteau_merged.csv,l star noomit nobasel r2 aic bic numbers replace