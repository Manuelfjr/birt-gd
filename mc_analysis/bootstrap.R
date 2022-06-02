#install.packages('nptest')
library('nptest')


# Datasets ###############
lista = list(
generate_data_mc100_m50_i300_e10000_t1000_lr1.0,
generate_data_mc100_m20_i100_e10000_t1000_lr1.0,
generate_data_mc100_m100_i100_e10000_t1000_lr1.0,

stats_results_df_mc_ALL_i300_m50,
stats_results_df_mc_ALL_i100_m20,
stats_results_df_mc_ALL_i100_m100)

names_bd = c('generate_data_mc100_m20_i100_e10000_t1000_lr1.0',
             'generate_data_mc100_m100_i100_e10000_t1000_lr1.0',
             'generate_data_mc100_m50_i300_e10000_t1000_lr1.0',
              'stats_results_df_mc_ALL_i100_m20',
              'stats_results_df_mc_ALL_i100_m100',
              'stats_results_df_mc_ALL_i300_m50')

names = c('thi','delj','aj')

# RSE - mu ##############
set.seed(2022)
for (j in 1:6){
  print(names_bd[j])
for (i in 1:length(names)){
  data = as.data.frame(lista[[j]])
  values = lista[[j]][,paste("RSE_",names[i],sep='')]
  npbs.media <- np.boot(x = values, statistic = mean)
  ics.bca = round(npbs.media$bca,4)
  print(paste('Param:',names[i]))
  print(ics.bca[2,])
  print(mean(values))
}
  print('')
}

# Rho - mu ##############
set.seed(2022)
for (j in 1:6){
  print(names_bd[j])
  for (i in 1:length(names)){
    data = as.data.frame(lista[[j]])
    values = lista[[j]][,paste("corr_",names[i],"_to_pred_",names[i],sep='')]
    npbs.media <- np.boot(x = values, statistic = mean)
    ics.bca = round(npbs.media$bca,4)
    print(paste('Parametro:',names[i]))
    print(ics.bca[2,])
    print(mean(values))
  }
  print('')
}