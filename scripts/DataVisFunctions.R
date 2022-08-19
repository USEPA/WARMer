## Data-Vis Functions
require("magrittr")  # needed for pipes (when dplyr et al. are not imported)
require("ggplot2")

#' Label WARM treatment sectors based on pathway substrings
#' @param df dataframe with a "sector" column of sector names
#' @return A dataframe of direct and per-tier-1-purchase sector impacts
labelSectorPathwaysAndMaterials <- function(df) {
  df <- df %>% dplyr::mutate(
    sector_pathway = dplyr::case_when(
      stringr::str_detect(sector, "Anaerobic digestion") ~ "AD",
      stringr::str_detect(sector, "MSW composting")      ~ "Compost",
      stringr::str_detect(sector, "MSW combustion")      ~ "Combustion",
      stringr::str_detect(sector, "MSW landfilling")     ~ "Landfill",
      stringr::str_detect(sector, "MSW recycling")       ~ "Recycle",
      TRUE ~ sector),  # else return original sector name str
    sector_material = stringr::str_extract(
      sector, "(?<= of ).*(?=;)|(?<= of ).*")
  )
  return(df)
}

#' Plot direct and per-tier-1-purchase impacts for all WARM treatment sectors
#' related to a specified material (i.e., a substring of the sector names)
#' @param df dataframe
#' @param material str, name of a material from WARM sector names
#' @param source_mod str, name of model results source
#' @return A stacked bar chart of direct and per-tier-1-purchase impacts by sector
plotByMaterial <- function(df, material, impact, source_mod) {
  if (missing(source_mod)) {
    df_plot <- df %>% 
      dplyr::filter(sector_material == material) %>% 
      dplyr::mutate(sector_pathway = paste0(sector_pathway, ", ", source))
  }
  else {
    df_plot <- df %>% 
      dplyr::filter(sector_material == material, 
                    source == source_mod)
  }
  
  df_totals <- df_plot %>% 
    dplyr::group_by(sector_pathway) %>% 
    dplyr::summarise(impact_per_purchase=sum(impact_per_purchase))

  p_summ = summary(df_plot$impact_per_purchase)  # summarize all impact values
  # create array of geom_text nudges via comparison to central plot value
  p_nudge_abs = (p_summ[["Max."]] - p_summ[["Min."]])/100
  p_nudge_mid = (p_summ[["Max."]] - p_summ[["Min."]])/2 + p_summ[["Min."]]
  p_nudge_dir = ifelse(df_totals$impact_per_purchase >= p_nudge_mid, -1, 1)
  p_nudge = p_nudge_abs * p_nudge_dir
  p_nudge = ifelse(df_totals$impact_per_purchase <0, p_nudge * 2, p_nudge)
    # extra space for minus signs
  
  impact_label <- setNames(c("kg CO2e", "USD 2012", "USD 2012", "Jobs"), 
                           c("Greenhouse Gases", "Wages", "Taxes", "Jobs Supported"))
  
  p <- df_plot %>% 
    ggplot() + theme_bw() +
    geom_col(aes(x=impact_per_purchase, y=sector_pathway, 
                 fill=purchased_commodity)) +
    geom_point(data=df_totals, aes(x=impact_per_purchase, y=sector_pathway)) +
    geom_text(data=df_totals, hjust="inward", size=3.5,
              nudge_x=p_nudge, 
              aes(x=impact_per_purchase, y=sector_pathway, 
                  label=formatC(impact_per_purchase, digits=3))) +
    theme(legend.position="bottom",
          legend.text=element_text(size=9)) +
    guides(fill=guide_legend(nrow=2, byrow=TRUE)) +  # wrap legend entries
    scale_y_discrete(limits=rev) +
    labs(fill="", #title=material, 
         x=paste0(impact, " [", impact_label[[impact]], " / kg ", material, "]"),
         y="Sector Pathway")
  return(p)
}


#' # [later] paste together units from model$Indicators values?
#' # [later] any value in converting certain string fields to factors?
