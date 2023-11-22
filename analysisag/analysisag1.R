library(tidyverse)
#files = list.files("../data", pattern = ".csv")
files = "../data/207844_ala-gemma_2023-11-22_19h48.17.643.csv"
allfiles=tibble()
for (f in files){
  fin=read_csv(show_col_types = F, f)
  fin=fin %>% filter(!is.na(trial_type))
  allfiles <- allfiles %>% bind_rows(fin)
}
keepfiles = allfiles %>% 
  select(participant, pairs.thisN, blocks.thisN, trial_count,
         match_label, match_shape,
         #mismatch_label, mismatch_shape,
         trial_type,
         trial_label, trial_shape,
         trial_response.keys, trial_response.rt,
         acc) %>% 
  mutate(pairs.thisN=pairs.thisN+1) %>% 
  mutate(blocks.thisN=blocks.thisN+1) %>% 
  mutate(trial_count=trial_count+1) %>% 
  rename(pair=pairs.thisN) %>% 
  rename(block=blocks.thisN) %>% 
  rename(trial=trial_count) %>% 
  rename(response=trial_response.keys) %>% 
  rename(RTms=trial_response.rt) %>% 
  mutate(RTms=RTms*1000)

write_csv(keepfiles, "output.csv")
