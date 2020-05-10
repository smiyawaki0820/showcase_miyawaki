cheatsheet do
    title 'R commands'
    docset_file_name 'cheat R'
    keyword 'R'
  
    category do
      id 'compatible with python'
  
      entry do
        name 'pyper'
        notes <<-'CODE'
          ```python
          pip install pyper
          ```
          ```python
          import pyper
          import pandas as pd
          df = pd.read_csv()
          r = pyper.R(use_pandas='True')  # create R instance
          r.assign('data', df)            # Pass Python object to R
          r("source(file='hoge.R')")
          ```
        CODE
      end
    end

    category do
      id '基本操作'
  
      entry do
        name '基本操作'
        notes <<-'CODE'
          ```R
          x <- 1:3            # 1, 2, 3
          y <- c(1,2,3)       # vector
          x * y               # 1, 4, 9

          plot(x, y)
          ```
        CODE
      end

      entry do
        name 'my関数'
        notes <<-'CODE'
          ```R
          func <- function() {
              return 0
          }
          ```
        CODE
      end
    
      entry do
        name '関数'
        notes <<-'CODE'
          ```R
          ?sort
          ```
          ```R
          sqrt(x)
          mean(x)
          floor(x)
          sort(c, decreasing=FALSE)
          ```
        CODE
      end
    end

    category do
        id 'stringr'
        # src https://github.com/rstudio/cheatsheets/blob/master/strings.pdf
        entry do
          name '文字列操作'
          notes <<-'CODE'
            ```R
            install.packages("stringr")
            library(stringr)
            ```
          CODE
        end

        entry do
            name '操作'
            notes <<-'CODE'
              ```R
              str_length(s)
              ```
              ```R
              str_split(s, pattern="")
              str_c(s1, s2, sep="")
              ```
              ```R
              str_count(s, pattern="")
              str_sub(t, start=1, end=-1)
              str_replace(s, pattern="", replacement="")
              ```
              ```R
              str_pad(s, width=, side="both", pad="@")
              str_trim(s, side="both")
              ```
            CODE
        end
        
        

    end
  
    category do
      id 'dplyer'
  
      entry do
        name 'Pandas'
        notes <<-'CODE'
          ```bash
          $ git stash
          ```
        CODE
      end
    end

    category do
        id 'ggplot2'
        entry do
          name 'Matplotlib'
          notes <<-'CODE'
            ```bash
            $ git stash
            ```
          CODE
        end
    end

    category do
        id 'caret'
        entry do
          name 'Scikit-learn'
          notes <<-'CODE'
            ```bash
            $ git stash
            ```
          CODE
        end
    end


  end