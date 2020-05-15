library(ggplot2)
library(formattable)
library(dplyr)
library(tidyverse)
library(plotly)
library(Rmisc)

# データ構造の確認
str(iris)
formattable(iris[c(1:5),])


### ggplot ###
# 1. ggplot: prepare canvas
# 2. geom: overwrite graphs
# 3. theme: set details


## 散布図
gscat <- ggplot(iris, 
       aes(x=Sepal.Length, y=Sepal.Width))+
  geom_point(
    mapping=aes(colour=Species, shape=Species, size=Petal.Width),
    )+ # 散布図
  geom_smooth() # 回帰線
  #theme_bw() # 白基調 grid


## ヒストグラム
ghist <- ggplot(iris,
       aes(x=Petal.Length, y=..density.., fill=Species))+
  geom_histogram(position='dodge',
                 bins=15,
                 binwidth=1.0)+
  geom_density(alpha=0.5)+ # transparency
  geom_vline( # set vline=0, linetype:solid
    xintercept=0, linetype='solid', size=1.0)+
  geom_hline(
    yintercept=1, linetype='dotted', size=1.0)+
  theme(
    plot.subtitle = element_text(
      family='serif', size=12, vjust=1),
    plot.caption = element_text(
      family='serif', vjust=1),
    axis.line = element_line(
      linetype='solid'),
    axis.ticks = element_line(
      colour='gray18'),
    panel.grid.major = element_line(
      colour='gray74'),
    panel.grid.minor = element_line(
      colour='gray93'),
    axis.title = element_text(
      family='serif', size=16),
    axis.text = element_text(
      family='serif', size=16, angle=0),
    plot.title = element_text(
      family='serif', size=18, hjust=0.5),
    panel.background = element_rect(
      fill='white'),
    plot.background = element_rect(
      fill='white')
  )+
  labs(
    title = 'Iris Histogram',
    x = 'Petal.Length',
    y = 'count',
    subtitle = 'ggThemeAssist',
    caption = 'Designed by Mr.Unadon: https://mrunadon.github.io/images/geom_kazutanR.html'
  )
  
  
## 折れ線グラフ

# データ作成
df <- data.frame(Index=as.array(c(1:10)),
                 Rand=rgamma(10, 1, 1))%>%
  dplyr::mutate(Lower=Rand-0.5,
                Upper=Rand+0.5)%>%
  dplyr::arrange(desc(Index)) # desc sort

# summary(df)
df%>%
  dplyr::summarise_each(
    funs(min, max, mean),
    Rand, contains('Index'))

formattable(df)

# plot
gline <- ggplot(df, aes(x=Index, y=Rand))+
  geom_line()+
  geom_ribbon( # display band
    aes(ymin=Lower, ymax=Upper),
    alpha=0.1)


## using plotly
ggplotly(gline)

## multiplot
gmul <- multiplot(
  gscat, ghist, gline,
  cols=2)

ggsave(file='sample.ggplot2.png', 
       plot=gmul,
       dpi=100,
       width=6.4, height=4.8)
