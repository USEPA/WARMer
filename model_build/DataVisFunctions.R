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
#' @return A stacked bar chart of direct and per-tier-1-purchase impacts by sector
plotByMaterial <- function(df, material) {
  df_plot <- df %>% dplyr::filter(sector_material == material)
  
  df_totals <- df_plot %>% 
    dplyr::group_by(sector_pathway) %>% 
    dplyr::summarise(impact_per_purchase=sum(impact_per_purchase))
  
  p_summ = summary(df_plot$impact_per_purchase)  # summarize all impact values
  p_nudge = (p_summ["Max."] - p_summ["Min."])/100  # to get a geom_text position adj.
  
  p <- df_plot %>% 
    ggplot() + theme_bw() +
    geom_col(aes(x=impact_per_purchase, y=sector_pathway, 
                 fill=purchased_commodity)) +
    geom_point(data=df_totals, aes(x=impact_per_purchase, y=sector_pathway)) +
    geom_text(data=df_totals, hjust="left", nudge_x=p_nudge, size=3.5,
              aes(x=impact_per_purchase, y=sector_pathway, 
                  label=round(impact_per_purchase, digits=3))) +
    theme(legend.position="bottom",
          legend.text=element_text(size=9)) +
    guides(fill=guide_legend(nrow=2, byrow=TRUE)) +  # wrap legend entries
    scale_y_discrete(limits=rev) +
    labs(title=material, fill="",
         x=paste0(impact, " [kg CO2e / kg ", material, "]"),
         y="Sector Pathway")
  return(p)
}

#' # [later] paste together units from model$Indicators values?
#' # [later] any value in converting certain string fields to factors?
#' 


## Using magrittr pipes:
# disaggregateTotalToDirectAndTier1 <- function(model, indicator) {
#   sector_map <- setNames(model$Commodities$Name, model$Commodities$Code_Loc)
#   
#   # direct sector impacts
#   df_A_D <- tibble::enframe(model$D[impact,]) %>%
#     dplyr::rename(impact_per_purchase=value, sector_code=name) %>% 
#     dplyr::mutate(purchased_commodity = 'Direct') # not a commodity, but for convenience
#   
#   # total impacts per Tier 1 purchase by sector
#   df_A_p <- calculateTotalImpactbyTier1Purchases(model, impact) %>% 
#     tibble::as_tibble(rownames="purchased_commodity_code") %>%
#     reshape2::melt(id.vars="purchased_commodity_code",
#                    variable.name="sector_code", 
#                    value.name="impact_per_purchase") %>%
#     dplyr::mutate(purchased_commodity = dplyr::recode(purchased_commodity_code, !!!sector_map))
#   
#   # combined tibble 
#   df_A <- dplyr::bind_rows(df_A_p, df_A_D) %>%
#     dplyr::mutate(sector = dplyr::recode(sector_code, !!!sector_map))
#   return(df_A)
# }
# ^magrittr::`%>%` pipe namespace issue if migrating to useeior; may need to refactor
