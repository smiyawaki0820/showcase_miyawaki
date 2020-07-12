export PATH="/home/miyawaki_shumpei/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PYTHONPATH="/work01/miyawaki/allennlp/seq2seq/configs:$PATH"


#### sbt, jdk

function setjdk {
    pathtojava=$(readlink -e $(which javac))
    export JAVA_HOME=${pathtojava%/*/*}
}
setjdk

path=( ~/jdk1.8.0_191/bin $path)

####

export PATH="/home/miyawaki_shumpei/local/bin/jdk1.8.0_191/bin:$PATH"  # jdk
export PATH="/home/miyawaki_shumpei/sh:$PATH"
export PATH="/home/miyawaki_shumpei/local/sbt/bin:$PATH"  # sbt
export PATH="/home/miyawaki_shumpei/sbt/bin:$PATH"  # sbt
export PATH="/home/miyawaki_shumpei/local/bin/jumanpp-2.0.0-rc3/bin:$PATH"
export PATH="/home/miyawaki_shumpei/showcase_miyawaki/yans_pack:$PATH"

export PERL5LIB="/home/miyawaki_shumpei/soft/srlconll-1.1/lib:$PERL5LIB"
export PATH="$HOME/soft/srlconll-1.1/bin:$PATH"

# Echo customized prompt
function echo-prompt() {
  echo "%{$fg_bold[green]%}${HOST} %F{blue}ms%f : %F{magenta}%~%f
%# "
}
setopt prompt_subst  # プロンプトが表示されるたびにプロンプト文字列を評価、置換する
#TMOUT=1
#TRAPALRM() #{zle reset-prompt}

PROMPT='`echo-prompt`'
RPROMPT='%F{black}%*'
#RPROMPT='%F{white} %D{%Y-%m-%d %H:%M:%S} %f'

autoload -Uz colors
colors


# -----------------------------
# ssh connection then exec tmux
# -----------------------------
clear 
cd /work01/miyawaki/exp/miya-fairseq-gec
nvidia-smi

#if [[ ! -n $TMUX && $- == *l* ]]; then
#  clear
#  tmux ls | awk '{print length() ,$0}' | sort -n | sed -r "s/[1-9]*(.*):.*?(\(.*\)).*?$/\2\t\1/" 
#  # get the IDs
#  ID="`tmux list-sessions`"
#  if [[ -z "$ID" ]]; then
#    tmux new-session
#  fi
#  read ID
#  if [[ -n "$ID" ]]; then
#    tmux attach-session -t "$ID" || tmux new-session -s "$ID"
#  else
#  #  :  # Start terminal normally
#  fi
#fi


# -----------------------------
# alias
# -----------------------------
alias runc='(){gcc -W $1 && ./a.out}'
alias b2="cd ../../"
alias b3="cd ../../../"
alias b4="cd ../../../../"
alias showcase="cd ~/cl-tohoku/showcase_miyawaki/"
alias imi="cd ~/cl-tohoku/Club-IMI-taiwa-2019/"
alias bert="cd ~/cl-tohoku/Club-IMI-taiwa-2019/sentiment_analysis/Japanese-sentiment-analysis/bert/"
alias gec="cd ~/cl-tohoku/miya-fairseq-gec/"

alias add='git add'
alias commit='git commit -m'
alias push='git push origin master'
alias branch="git branch"
alias checkout="git checkout"
alias url="git config --get remote.origin.url"
alias readme="vim README.md"
alias status="git status --short"
alias log="git log --oneline"
alias push_ignore="git add .gitignore && git commit -m 'update .gitignore'"
alias push_readme="git add README.md && git commit -m 'update README.md'"
alias grm="git rm --cached"

alias tm="tmux new-session -s"
alias ta="tmux a -t"
alias tk="tmux kill-session -t"
alias tls="tmux ls"

alias run="bash run.sh"
alias train="bash train.sh"
alias test="bash test.sh"

alias py="python"
alias ipy="ipython"

alias vv="vim ~/.vimrc"
alias e="exit"
alias c='cd ./'
alias gpu='nvidia-smi'
alias cpu='ps aux --sort -%cpu | head -10'
alias memory='free -m'
alias sz="source ~/.zshrc"
alias vz="vim ~/.zshrc"
alias mkdir="mkdir -p"
alias ls="ls -XF"
alias cp="cp -v"

export WSJ="/home/miyawaki_shumpei/cl-tohoku/miya-fairseq-gec/datasets/conll05/conll05.test.wsj.prop"
alias wsj="perl ~/soft/srlconll-1.1/bin/srl-eval.pl ${WSJ} "

alias test0='bash test.sh -g 0 -i ~/PAS/NTC_Matsu_converted'
alias test1='bash test.sh -g 1 -i ~/PAS/NTC_Matsu_converted'
alias test2='bash test.sh -g 2 -i ~/PAS/NTC_Matsu_converted'
alias train0='bash train_edit.sh -g 0 -i ~/PAS/NTC_Matsu_converted'
alias train1='bash train_edit.sh -g 1 -i ~/PAS/NTC_Matsu_converted'
alias train2='bash train_edit.sh -g 2 -i ~/PAS/NTC_Matsu_converted'
alias e0='bash exec.sh -g 0 -i ~/PAS/NTC_Matsu_converted'
alias e1='bash exec.sh -g 1 -i ~/PAS/NTC_Matsu_converted'
alias e2='bash exec.sh -g 2 -i ~/PAS/NTC_Matsu_converted'
alias cdwork="cd && cd ../../work01/miyawaki"
alias cdbert="cd ~/cl-tohoku/Club-IMI-taiwa-2019/Japanese-sentiment-analysis/src/bert"
alias cdallen="cd && cd ../../work01/miyawaki/QA-SRL/SRL_S2S"
alias evalpas="cd ~/Club-IMI-taiwa-2019/test/evalpas"
alias edit='vim ./src/train_edit.py'
alias eva='vim ./src/eval_edit.py'
alias model="vim src/model_edit.py"

alias adnis00='ssh miyawaki_shumpei@adnis00'
alias adnis01='ssh miyawaki_shumpei@adnis01'
alias adnis02='ssh miyawaki_shumpei@adnis02'
alias ama01='ssh miyawaki_shumpei@amaretto01'
alias ama02='ssh miyawaki_shumpei@amaretto02'
alias neg01='ssh miyawaki_shumpei@negroni01'
alias neg02='ssh miyawaki_shumpei@negroni02'
alias spu01='ssh miyawaki_shumpei@spumoni01'
alias spu02='ssh miyawaki_shumpei@spumoni02'
alias spu04='ssh miyawaki_shumpei@spumoni04'
alias spu05='ssh miyawaki_shumpei@spumoni05'


if [[ -x `which colordiff` ]]; then # installされていたら
  alias diff='colordiff'
else
  alias diff='diff'
fi

chpwd() {
  clear
  echo `ls -lt | grep ^d | awk '{print $9}'`
  PY=`ls -XF -lt | awk '{print $9}' | grep .py | grep -v / | head -10 | xargs`
  SH=`ls -XF -lt | awk '{print $9}' | grep .sh | grep -v / | head -10 | xargs`
  if [ -n "${PY}" ] ; then
    echo -en "\n[py] `echo ${PY}`\n"
  fi
  if [ -n "${SH}" ] ; then 
    echo -en "[sh] `echo ${SH}`\n"
  fi 
}


# -----------------------------
# Completion
# -----------------------------

autoload -Uz compinit ; compinit # 自動補完を有効にする
#setopt complete_in_word #単語の入力途中でもTab補完を有効か

# コマンドミスを修正
setopt correct

zstyle ':completion:*' menu select # 補完の選択を楽にする

# 補完候補をできるだけ詰めて表示する
setopt list_packed
# 補完候補にファイルの種類も表示する
setopt list_types

# -----------------------------
# git
# -----------------------------

# ブランチ名を色付きで表示させるメソッド
function rprompt-git-current-branch {
  local branch_name st branch_status

  if [ ! -e  ".git" ]; then
    # gitで管理されていないディレクトリは何も返さない
    return
  fi
  branch_name=`git rev-parse --abbrev-ref HEAD 2> /dev/null`
  st=`git status 2> /dev/null`
  if [[ -n `echo "$st" | grep "^nothing to"` ]]; then
    # 全てcommitされてクリーンな状態
    branch_status="%F{green}"
  elif [[ -n `echo "$st" | grep "^Untracked files"` ]]; then
    # gitに管理されていないファイルがある状態
    branch_status="%F{red}?"
  elif [[ -n `echo "$st" | grep "^Changes not staged for commit"` ]]; then
    # git addされていないファイルがある状態
    branch_status="%F{red}+"
  elif [[ -n `echo "$st" | grep "^Changes to be committed"` ]]; then
    # git commitされていないファイルがある状態
    branch_status="%F{yellow}!"
  elif [[ -n `echo "$st" | grep "^rebase in progress"` ]]; then
    # コンフリクトが起こった状態
    echo "%F{red}!(no branch)"
    return
  else
    # 上記以外の状態の場合は青色で表示させる
    branch_status="%F{blue}"
  fi
  # ブランチ名を色付きで表示する
  echo "${branch_status}[$branch_name]"
}

# プロンプトが表示されるたびにプロンプト文字列を評価、置換する
setopt prompt_subst

# プロンプトの右側(RPROMPT)にメソッドの結果を表示させる
RPROMPT='`rprompt-git-current-branch` %F{black}%*'

# -----------------------------
# History
# -----------------------------
HISTFILE=~/.zsh_history
HISTSIZE=6000000
SAVEHIST=6000000
setopt share_history

[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh
export FZF_DEFAULT_OPTS='--height 40% --reverse --border'
fzcd() {
   local dir
   dir=$(find ${1:-.} -path '*/\.*' -prune \
                  -o -type d -print 2> /dev/null | fzf +m) &&
   cd "$dir"
}

# fbr - checkout git branch
fbr() {
  local branches branch
  branches=$(git branch -vv) &&
  branch=$(echo "$branches" | fzf +m) &&
  git checkout $(echo "$branch" | awk '{print $1}' | sed "s/.* //")
}

# fshow - git commit browser
fshow() {
  git log --graph --color=always \
      --format="%C(auto)%h%d %s %C(black)%C(bold)%cr" "$@" |
  fzf --ansi --no-sort --reverse --tiebreak=index --bind=ctrl-s:toggle-sort \
      --bind "ctrl-m:execute:
                (grep -o '[a-f0-9]\{7\}' | head -1 |
                xargs -I % sh -c 'git show --color=always % | less -R') << 'FZF-EOF'
                {}
FZF-EOF"
}

# fd - cd to selected directory
fd() {
  local dir
  dir=$(find ${1:-.} -path '*/\.*' -prune \
                  -o -type d -print 2> /dev/null | fzf +m) &&
  cd "$dir"
}
export SA_DATA='/work01/miyawaki/data/twitter_data_for_sentiment'

source ~/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh
