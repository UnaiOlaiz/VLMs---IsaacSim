%% templateDeusto.cls (Official PFG Template 2024–25 - University of Deusto)
%% Fully translated to English

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{templateDeusto}[2025/03/26 v0.5 Adapted to Deusto University PFG Regulations]

%% ------------------------------
%% Base Class Loading
%% ------------------------------
\DeclareOption{draft}{\PassOptionsToPackage{draft}{graphicx}}
\ProcessOptions\relax
\LoadClass[10pt,a4paper,oneside,open=any]{scrbook} % KOMA-Script Base

%% ------------------------------
%% Essential Packages
%% ------------------------------
\RequirePackage[automark,markcase=upper]{scrlayer-scrpage} % Headers/Footers
\RequirePackage[T1]{fontenc}      % Font encoding
\RequirePackage[utf8]{inputenc}    % Input encoding
\RequirePackage[english]{babel}    % Language: English
\RequirePackage{setspace}          % Line spacing
% \RequirePackage[default]{sourcesanspro} % Modern sans-serif font
% \renewcommand{\familydefault}{\sfdefault}
\RequirePackage{url}               % URLs
\RequirePackage{fancybox}          % Boxes
\RequirePackage[pdftex]{graphicx}  % Images
\RequirePackage{ifthen}            % Logic
\RequirePackage[pdftex]{color}     % Colors
\RequirePackage{fancyvrb}          % Verbose
\RequirePackage{amssymb}           % Math symbols
\RequirePackage{multibib}          % Multiple bibliographies
\RequirePackage{listings}          % Source code
\RequirePackage{pdfpages}          % Include PDFs
\RequirePackage[colorlinks=true,
linkcolor=black,
citecolor=blue,
urlcolor=blue,
unicode,
hypertexnames=false,
bookmarks=true]{hyperref}
\RequirePackage{bookmark}          % PDF Improvements
\RequirePackage{xcolor} 
\RequirePackage{amsmath}
\RequirePackage{float}
\RequirePackage{array}
\RequirePackage{tikz}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{caption}
\RequirePackage{pgfgantt}
\RequirePackage{rotating}
\RequirePackage{xurl}


% TO NOT RAISE CHINESE ERRORS
\DeclareUnicodeCharacter{5347}{\relax}
\DeclareUnicodeCharacter{7EA7}{\relax}
\DeclareUnicodeCharacter{4E4B}{\relax}
\DeclareUnicodeCharacter{8DEF}{\relax}
\DeclareUnicodeCharacter{FF1A}{\relax}
\DeclareUnicodeCharacter{3001}{\relax}
\DeclareUnicodeCharacter{4E8C}{\relax}
\DeclareUnicodeCharacter{95F4}{\relax}
\DeclareUnicodeCharacter{7A7A}{\relax}
\DeclareUnicodeCharacter{79D1}{\relax}
\DeclareUnicodeCharacter{5B66}{\relax}
\DeclareUnicodeCharacter{7801}{\relax}
\DeclareUnicodeCharacter{7F16}{\relax}
\DeclareUnicodeCharacter{7F6E}{\relax}
\DeclareUnicodeCharacter{4F4D}{\relax}
\DeclareUnicodeCharacter{5F0F}{\relax}
\DeclareUnicodeCharacter{8F6C}{\relax}
\DeclareUnicodeCharacter{65CB}{\relax}
\DeclareUnicodeCharacter{7684}{\relax}
\DeclareUnicodeCharacter{7EF4}{\relax}
%% ------------------------------
%% Listings Configuration (Code)
%% ------------------------------
\definecolor{deustoBlue}{cmyk}{1,0.57,0,0.02}
\definecolor{graybg}{gray}{0.97}
\definecolor{commentgray}{gray}{0.4}
\definecolor{stringorange}{rgb}{0.8,0.4,0.1}

\usepackage{tikz}
\usetikzlibrary{positioning, fit, backgrounds, calc}

\lstset{
	backgroundcolor=\color{graybg},
	basicstyle=\ttfamily\small,
	keywordstyle=\color{gray}\bfseries, %deustoBlue
	commentstyle=\color{commentgray}\itshape,
	stringstyle=\color{black}, %stringorange
	numberstyle=\tiny\color{gray},
	numbers=left,
	numbersep=10pt,
	frame=single,
	framesep=5pt,
	rulecolor=\color{black},
	breaklines=true,
	showstringspaces=false,
	tabsize=2,
	captionpos=b,
	xleftmargin=1em,
	xrightmargin=1em,
	aboveskip=1em,
	belowskip=1em,
	extendedchars=true,
	inputencoding=utf8
}

%% ------------------------------
%% Metadata & Format Definitions
%% ------------------------------
\makeatletter
\newcommand{\@authorOne}{Author Name}
\newcommand{\@authorTwo}{}
\newcommand{\@authors}{\@authorOne}
\newcommand{\authors}[2]{\def\@authorOne{#1}\def\@authorTwo{#2}}

\newcommand{\degree}[1]{\def\@degree{#1}}
\newcommand{\supervisor}[1]{\def\@supervisor{Supervisor: #1}}

\gdef\@abstractText{}
\gdef\@keywordsText{}
\newcommand{\abstractcontent}[1]{\gdef\@abstractText{#1}}
\newcommand{\keywords}[1]{\gdef\@keywordsText{#1}}

\AtBeginDocument{
	\ifthenelse{\equal{\@authorTwo}{}}{\def\@authors{\@authorOne}}{\def\@authors{\@authorOne{} and \@authorTwo}}
	\pagestyle{scrheadings}
	\setstretch{1.5}           % 15pt spacing for 10pt body
	\setlength{\parskip}{12pt}  % Paragraph spacing
	\setlength{\parindent}{0pt} % No indent
	\hypersetup{
		linktoc=all,
		pdftitle={\@title},
		pdfauthor={\@authors},
		pdfsubject={Final Degree Project},
		pdfkeywords={\@keywordsText}
	}
}
\makeatother

%% ------------------------------
%% Margins & Layout
%% ------------------------------
\RequirePackage[a4paper,
top=1.0cm,
headheight=0.2cm,
headsep=2.4cm,
bottom=1.2cm,
footskip=3.3cm,
inner=2.5cm,
outer=2.5cm,
includehead,
includefoot
]{geometry}

%% ------------------------------
%% Headers & Footers
%% ------------------------------
\clearpairofpagestyles
\ihead{}
\chead{}
\ohead{{\fontsize{8}{10}\selectfont\textit{FINAL DEGREE PROJECT}}}

\ifoot{}
\cfoot{}
\ofoot{\raisebox{6pt}{{\fontsize{11}{13}\selectfont\pagemark}}}

%% ------------------------------
%% Document Structure
%% ------------------------------
\renewcommand{\frontmatter}{%
	\pagestyle{scrheadings}
	\pagenumbering{roman}
}

\renewcommand{\mainmatter}{%
	\cleardoublepage
	\pagestyle{scrheadings}
	\pagenumbering{arabic}
	\setcounter{secnumdepth}{5}
	\setcounter{tocdepth}{3}
	
	%% Chapters: 16pt, Uppercase, Underlined
	\RedeclareSectionCommand[
	beforeskip=0pt,
	afterskip=0.5ex,
	font=\bfseries\fontsize{16}{18}\selectfont,
	innerskip=0pt
	]{chapter}
	\addtokomafont{chapter}{\MakeUppercase}
	\renewcommand*{\chapterformat}{\thechapter.\enskip}
	\renewcommand*{\chapterlinesformat}[3]{%
		\hfill##2##3\par
		\vspace*{-3ex}
		\noindent\rule{\textwidth}{1pt}%
	}
	
	%% Sections: 14pt
	\RedeclareSectionCommand[
	beforeskip=1.5ex plus .2ex,
	afterskip=1ex plus .1ex,
	font=\bfseries\fontsize{14}{16}\selectfont
	]{section}
	\addtokomafont{section}{\MakeUppercase}
	
	%% Subsections: 12pt
	\RedeclareSectionCommand[
	beforeskip=1.2ex plus .2ex,
	afterskip=0.8ex plus .1ex,
	font=\large\bfseries
	]{subsection}
	
	\RedeclareSectionCommand[
	beforeskip=1ex,
	afterskip=0.5ex,
	font=\normalsize\bfseries
	]{subsubsection}
}

\renewcommand{\backmatter}{%
	\cleardoublepage
	\pagestyle{scrheadings}
	\RedeclareSectionCommand[
	beforeskip=0pt,
	afterskip=0.5ex,
	font=\bfseries\fontsize{16}{18}\selectfont,
	innerskip=0pt
	]{chapter}
	\addtokomafont{chapter}{\MakeUppercase}
}

%% ------------------------------
%% Custom Commands (English)
%% ------------------------------
\newcommand{\customAbstract}[2]{%
	\cleardoublepage
	\pagenumbering{roman}
	\pagestyle{headings}
	\vspace*{3cm}
	{\bfseries\Large Abstract\par}
	\vspace{0.5cm}
	#1 
	\vspace{2cm}
	
	{\bfseries\Large Index terms\par}
	\vspace{0.5cm}
	#2 
	\cleardoublepage
}

\newcommand{\makeAbstract}{%
	\cleardoublepage\pagebreak
	{\Large\bfseries Abstract\par\vspace{1em}}%
	\noindent\@abstractText\par
	\vspace{2em}%
	{\Large\bfseries Keywords\par\vspace{1em}}%
	\noindent\@keywordsText\thispagestyle{plain}%
}

%% ------------------------------
%% Bibliography (IEEE)
%% ------------------------------
\newcites{others}{Bibliography}
\newcommand{\printBibliography}[1]{%
	\cleardoublepage
	\addcontentsline{toc}{chapter}{Bibliography}
	\renewcommand{\bibname}{BIBLIOGRAPHY}
	\bibliographystyle{IEEEtran}
	\bibliography{#1}
}

%% ------------------------------
%% Pseudocode & Paragraphs
%% ------------------------------
\renewcommand{\lstlistlistingname}{List of Pseudocodes}
\renewcommand{\lstlistingname}{Pseudocode}

\RedeclareSectionCommand[
beforeskip=1ex,
afterskip=0.5ex,
font=\normalsize\bfseries,
runin=false,
indent=0pt,
level=5
]{paragraph}
\renewcommand*{\paragraphformat}{\theparagraph.\enskip}

\RedeclareSectionCommand[runin=false]{subparagraph}