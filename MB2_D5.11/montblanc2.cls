\LoadClass[11pt, a4paper, twoside]{article}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{montblanc}[2012/03/20 Montblanc Project class]

\RequirePackage{color}
\RequirePackage{colortbl}
\RequirePackage{fancyhdr}
\RequirePackage{fullpage}
\RequirePackage{graphicx}
\RequirePackage{tabularx}
\RequirePackage[small]{titlesec}
\RequirePackage{xspace}
\RequirePackage{hyphenat}

% Title commands
\renewcommand{\title}[1]{\def\@title{#1}}
\newcommand{\devnum}[1]{\def\@devnum{#1}}
\newcommand{\version}[1]{\def\@version{#1}}
\newcommand{\deadline}[1]{\def\@deadline{#1}}
\newcommand{\level}[1]{\def\@level{#1}}
\newcommand{\nature}[1]{\def\@nature{#1}}
\newcommand{\authors}[1]{\def\@authors{#1}}
\newcommand{\contributors}[1]{\def\@contributors{#1}}
\newcommand{\reviewers}[1]{\def\@reviewers{#1}}
\newcommand{\keywords}[1]{\def\@keywords{#1}}

\renewcommand\maketitle{
    \thispagestyle{empty}
    \begin{figure}[h]
    \hspace{1.3cm}\includegraphics[width=14cm]{./montblanc-logo}
    \end{figure}

    \vspace{3cm}

    \begin{flushright}
    \Large\bf\color[rgb]{0.16, 0.43, 0.45}
    \textsf{D\@devnum -- \@title} \\
    \textsf{Version \@version}
    \end{flushright}

    \vspace{3cm}

    \begin{flushleft}
    \Large\bf\color[rgb]{0.16, 0.43, 0.45}
    \textsf{Document Information}
    \end{flushleft}

    \begin{table}[h]
    \begin{center}
    \begin{tabular}{|l|p{0.7\textwidth}|}
    \hline
    \color[rgb]{0.16, 0.43, 0.45}Contract Number & 610402 \\ \hline
    \color[rgb]{0.16, 0.43, 0.45}Project Website & www.montblanc-project.eu \\ \hline
    \color[rgb]{0.16, 0.43, 0.45}Contractual Deadline & M\@deadline \\ \hline
    \color[rgb]{0.16, 0.43, 0.45}Dissemination Level & \@level \\ \hline
    \color[rgb]{0.16, 0.43, 0.45}Nature & \@nature \\ \hline
    \color[rgb]{0.16, 0.43, 0.45}Authors & \@authors \\ \hline
    \color[rgb]{0.16, 0.43, 0.45}Contributors & \@contributors \\ \hline
    \color[rgb]{0.16, 0.43, 0.45}Reviewers & \@reviewers \\ \hline
    \color[rgb]{0.16, 0.43, 0.45}Keywords & \@keywords \\ \hline
    \end{tabular}
    \end{center}
    \end{table}

    {\footnotesize
    \em
    \textbf{Notices:}
    The research leading to these results has received funding from the
    European Community's Seventh Framework Programme (FP7/2007-2013) under
    grant agreement n$^o$ 610402.
    \\
    \\
    \copyright Mont-Blanc 2 Consortium Partners. All rights reserved.
    }
}

\newcommand{\frontmatter}{
    \newpage
    \vspace*{-0.9cm}
    \tableofcontents
    \newpage
}

\newenvironment{executive}{
    \newpage
    \section*{Executive Summary}
    \addcontentsline{toc}{section}{Executive Summary}
    }{
    \vfill
    \newpage
    }

\newenvironment{changelog}{
    \newpage
    \begin{flushleft}
    \Large\bf\color[rgb]{0.16, 0.43, 0.45}
    \vspace*{0.5cm}
    \textsf{Change Log}
    \end{flushleft}

    \begin{center}
    \begin{tabular}{|l|l|}
    \hline
    \rule[-0.3cm]{0mm}{0.8cm} \centering \bf \textsf{Version} & \centering \bf \textsf{Description of Change} \tabularnewline
    } {
    \hline
    \end{tabular}
    \end{center}
    \newpage
    }

\newcommand{\change}[2]{
    \hline
    \rule[-0.3cm]{0mm}{0.8cm} \centering v#1 & #2 \\ 
}

\titleformat{\section}[hang]{\huge\bfseries\sffamily\color[rgb]{0.16, 0.43, 0.45}}{\thesection}{1em}{}
\titleformat{\subsection}[hang]{\Large\bfseries}{\thesubsection}{1em}{}

\newcommand{\fixme}[1]{\textcolor{red}{\textit{FIXME $->$ #1 $<-$}}}
\newcommand{\todo}[1]{\textcolor{red}{\textbf{TODO: #1}}}
\newcommand{\eg}{e.g.,\xspace}
\newcommand{\ie}{i.e.,\xspace}
\newcommand{\etc}{etc.\@\xspace}
\newcommand{\etal}{et.\ al.\xspace}
\newcommand{\vs}{vs.\xspace}
\newcommand{\MB}{Mont\hyp{}Blanc\xspace}
\newcommand{\IO}{I\slash O\xspace}
\newcommand{\GFW}{GFLOPs\slash Watt\xspace}

\newcommand{\slogo}{\includegraphics[width=4cm]{./montblanc-logo}}

\AtBeginDocument{
    \headheight=20pt
    \pagestyle{fancy}
    \fancyhf{}
    \headsep 2ex
    \footskip 4ex
    \rhead[\slogo]{\slogo}
    \lhead[\scriptsize{D\@devnum} - {\@title}{\\Version \@version}]{\scriptsize{D\@devnum} - {\@title}{\\Version \@version}}
    \cfoot{\thepage}

    \renewcommand{\sectionmark}[1]{\markboth{#1}{}}
    \renewcommand{\subsectionmark}[1]{\markright{#1}{}}
}
