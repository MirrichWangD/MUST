
\NeedsTeXFormat{LaTeX2e}		[1995/12/01]
\ProvidesClass{format/MUSTThesisC}			[2019/05/27 v0.1.0]
\LoadClass[14pt, a4paper,openany]{ctexbook}		[2007/10/19]



\RequirePackage{lscape}
\RequirePackage{pdflscape}
\RequirePackage[top=2.5cm, bottom=2.5cm, left=2.5cm, right=2.5cm]{geometry}

%这里是版面设定 的页边距 A4 紙張，边界为(2.5cm)、下(2.5cm)、左(3.8cm)，右(2.5cm)，装订位置在左边。(供參考)

%% -------------------------------------------------<<加载必要的中文包,并指定字体
\RequirePackage{xeCJK}
\setCJKmainfont[AutoFakeSlant=0.3, AutoFakeBold=2.17,Path=fonts/]{KaiTi}
\setmainfont{Times New Roman}% 指定英文字體

\RequirePackage{fontspec}

%% -------------------------------------------------<< 加載開源宏包

\RequirePackage{format/fontawesome}


%% -------------------------------------------------<<加載自定义论文样式宏包
\RequirePackage{format/MUSTThesisS}





%% -------------------------------------------------<<中英文摘要格式
% 以上類似一個抽象函數
\newcommand{\@mustAbstract}[5]{
	\chapter*{\textbf{#1}}
	\markboth{#1}{}
	\addcontentsline{toc}{chapter}{#1}  %上一行在chapter后面加*，即让这章无编号，也不在目录中显示，加上这行后，在目录中显示
	\zihao{4}\linespread{#5}\selectfont{#2}
	\vspace{.8cm}
	\par\noindent\textbf{#3}#4
	\newpage
} 
% 外部調用命令(類似API) \mustCabstract \mustEabstract
\newcommand{\mustCabstract}[2]{\@mustAbstract{摘\quad 要}{#1}{關鍵詞：}{#2}{1.6}} % 生成中文摘要格式
\newcommand{\mustEabstract}[2]{\@mustAbstract{Abstract}{#1}{KeyWords：}{#2}{1.3}} % 生成英文摘要格式
%% ------------------------------------------------->>

 
%new list of figures
\newcommand{\@mustlistoffigures}[5]{
	\chapter*{\textbf{#1}}
	\markboth{#1}{}
	\addcontentsline{toc}{chapter}{#1}  
	\zihao{4}\linespread{#5}\selectfont{#2}
	\vspace{.8cm}
	\par\noindent\textbf{#3}#4
	\newpage
} 


\newcommand{\listfigures}[2]{\@mustlistoffigures{List of Figures}{#1}{}{#2}{1}} %
%new



%new list of figures
\newcommand{\@mustlistoftables}[5]{
	\chapter*{\textbf{#1}}
	\markboth{#1}{}
	\addcontentsline{toc}{chapter}{#1}  %上一行在chapter后面加*，即让这章无编号，也不在目录中显示，加上这行后，在目录中显示
	\zihao{4}\linespread{#5}\selectfont{#2}
	\vspace{.8cm}
	\par\noindent\textbf{#3}#4
	\newpage
} 


\newcommand{\listtables}[2]{\@mustlistoftables{List of Tables}{#1}{}{#2}{1}} %
%new

\newcommand{\@mustlistofsymbols}[5]{
	\chapter*{\textbf{#1}}
	\markboth{#1}{}
	\addcontentsline{toc}{chapter}{#1}  %上一行在chapter后面加*，即让这章无编号，也不在目录中显示，加上这行后，在目录中显示
	\zihao{4}\linespread{#5}\selectfont{#2}
	\vspace{.8cm}
	\par\noindent\textbf{#3}#4
	\newpage
} 


\newcommand{\listsymbols}[2]{\@mustlistofsymbols{List of Symbols}{#1}{}{#2}{1}} %
%new





%new list of abbreviations
\newcommand{\@listofabbreviations}[5]{
	\chapter*{\textbf{#1}}
	\markboth{#1}{}
	\addcontentsline{toc}{chapter}{#1}  %上一行在chapter后面加*，即让这章无编号，也不在目录中显示，加上这行后，在目录中显示
	\zihao{4}\linespread{#5}\selectfont{#2}
	\vspace{.8cm}
	\par\noindent\textbf{#3}#4
	\newpage
} 


\newcommand{\listabbreviations}[2]{\@listofabbreviations{List of Abbreviations}{#1}{}{#2}{1}} %
%new


%% -------------------------------------------------<<頁眉格式
% 设置页眉
\pagestyle{fancy}
% 頁眉左側顯示內容 \Ctitle 論文的題目
% \lhead{\fontsize{10pt}{\baselineskip}\selectfont\Etitle}
% 頁眉右側顯示內容
% \rhead{\fontsize{10pt}{\baselineskip}\selectfont\leftmark}
% 左侧标题设置（修正对齐）
\lhead{\parbox[b]{0.6\textwidth}{\raggedright\fontsize{10pt}{\baselineskip}\selectfont\Etitle}}
% 右侧章节标记（修正对齐）
\rhead{\parbox[b]{0.3\textwidth}{\raggedleft\fontsize{10pt}{\baselineskip}\selectfont\leftmark}}
% 定义页码显示形式
\pagenumbering{Roman}
% 定义此頁不顯示頁眉，但是頁腳中部顯示頁碼，plain替換為empty，則頁眉頁腳均不顯示
\pagestyle{fancy}
%% ------------------------------------------------->>






%% -------------------------------------------------<<公式標號
% 调整文中公式的變換
\makeatletter
\@addtoreset{equation}{chapter}
\makeatother  
\renewcommand\theequation{\arabic{chapter}.\arabic{equation}}
%% ------------------------------------------------->>






%% -------------------------------------------------<<目錄格式
\CTEXsetup[name={,},number={\arabic{chapter}}]{chapter}
\usepackage[titles]{tocloft}
\renewcommand{\cftdot}{$\cdot$}
\renewcommand{\cftdotsep}{1.5}
\setlength{\cftbeforechapskip}{0pt}
\renewcommand{\cftchapfont}{\zihao{4}}
\renewcommand{\cftchappagefont}{\zihao{4}}
\renewcommand{\cftchapleader}{\cftdotfill{\cftchapdotsep}}
\renewcommand{\cftchapdotsep}{\cftdotsep}
\makeatletter
\renewcommand{\numberline}[1]{%
	\settowidth\@tempdimb{#1\hspace{0.5em}}%
	\ifdim\@tempdima<\@tempdimb%
	\@tempdima=\@tempdimb%
	\fi%
	\hb@xt@\@tempdima{\@cftbsnum #1\@cftasnum\hfil}\@cftasnumb}
\makeatother
\allowdisplaybreaks[4]
\lstset{breaklines}
\renewcommand\thechapter{\arabic{chapter}}  %调整目录中的1 为中文下的第一章
\renewcommand\thesection{\arabic{chapter}.\arabic{section}}  %调整目录中的1.1为中文下的第一節   %调整
%% ------------------------------------------------->>
 
 
 
 
 
 
%% -------------------------------------------------<<設置參考文獻格式
\makeatletter
\renewenvironment{thebibliography}[1]
{\section*{\refname}%
	\@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
	\list{\@biblabel{\@arabic\c@enumiv}}%
	{\settowidth\labelwidth{\@biblabel{#1}}%
		\leftmargin\labelwidth
		\advance\leftmargin\labelsep
		\advance\leftmargin by 2em%
		\itemindent -2em%conclusion
		\@openbib@code
		\usecounter{enumiv}%
		\let\p@enumiv\@empty
		\renewcommand\theenumiv{\@arabic\c@enumiv}}%
	\sloppy
	\clubpenalty4000
	\@clubpenalty \clubpenalty
	\widowpenalty4000%
	\sfcode`\.\@m}
{\def\@noitemerr
	{\@latex@warning{Empty `thebibliography' environment}}%
	\endlist}
\makeatother
\setlength{\bibsep}{0.5ex}
%% ------------------------------------------------->>
 
 
 
 
 
 
%% -------------------------------------------------<<章節名格式   %修改章节的样式
\usepackage{titlesec}
\setcounter{secnumdepth}{4}
\renewcommand{\chaptername}{{\thechapter}}
\titleformat{\chapter}{\centering\bfseries\fontsize{22pt}{1pt}\selectfont}{Chapter \arabic{chapter}}{1em}{\thispagestyle{fancy}}[\vspace{-0.5cm}]
\titleformat{\section}{\fontsize{16pt}{1pt}\selectfont\bfseries}{\arabic{chapter}.\arabic{section}}{0.5em}{}[\vspace{-1cm}]
\titleformat{\subsection}{\fontsize{16pt}{1pt}\selectfont\bfseries}{\arabic{chapter}.\arabic{section}.\arabic{subsection}}{0.5em}{}
\titleformat{\subsubsection}{\fontsize{14pt}{1pt}\selectfont\bfseries}{\arabic{subsection}.\arabic{subsubsection}}{.5em}{}
\titleformat{\paragraph}{\fontsize{14pt}{\baselineskip}\selectfont}{\theparagraph}{.25em}{}
% 說明: \titlespacing*{章節命令} {左間距}{上間距}{下間距}[右間距]
\titlespacing{\chapter}{0pt}{-5pt}{50pt}
\titlespacing*{\section} {0pt}{40pt}{40pt}
\titlespacing*{\subsection}{0pt}{35pt}{10pt}
\titlespacing*{\subsubsection}{0pt}{20pt}{10pt}
\titlespacing*{\paragraph}{0pt}{5pt}{5pt}
% 设置文章段落不要弹性伸缩
\setlength\parskip{0pt}
%% ------------------------------------------------->>
 
 
 
 
 
 
%% -------------------------------------------------<<設置圖表序號
\numberwithin{figure}{chapter}
\numberwithin{table}{chapter}
\renewcommand{\thetable}{\arabic{chapter}.\arabic{table}}
\renewcommand{\thefigure}{\arabic{chapter}.\arabic{figure}}
%% ------------------------------------------------->>



 
 
\endinput