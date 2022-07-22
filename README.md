Dataset for `Accurate Mobile-App Fingerprinting Using Flow-level Relationship with Graph Neural Networks`.

# Intro

To evaluate the generalization of our method in dealing with ambiguous traffic and the performance against traffic concepts drift, we also collected another private encrypted mobile application traffic dataset across weeks.
Here go the details of the dataset setup.
\begin{itemize}
  \item \textbf{Equipment setup.}
  As Fig~\ref{fig1} indicates, smartphones with apps communicate with the Internet via a WiFi access point (AP) and the AP forwards the packets into two gateways that come from different ISPs.
  To generate traffic from apps, we used scripts that communicated with the target mobile via USB using Android Debug Bridge (ADB).
  These scripts were sent by the controller computer, and mainly contained UI commands that simulated user actions within apps and system commands that configured the devices.
\item \textbf{Applications selection.}
  We selected 53 apps from the apps list used by AppScanner after filtering out several plain-text applications which consist of a relatively low fraction of encrypted traffic.
  These apps come from different regions such as shopping, magazines, social and so on.
  We always installed the latest versions onto the selected devices and signed up for each app.
  \item \textbf{Network trace collection.}\footnote{
  Ethical Considerations: The network traces collected in this paper are all triggered by automatical UI fuzzing operations, thus no privacy information associated with human subjects would be leaked.}
  We cyclically performed UI fuzzing operations on each app which is activated for about 30 seconds every time via monkeyrunner as Appscanner~\cite{taylor2016appscanner} and passively collected the network traffic between smartphones and the AP.
  %Since Linux 2.6.14, it's possible to filter network packets by their owners with the help of iptables-extensions.
  To collect pure network traces from specific apps, we configured the android devices with the iptables rules and listened on the NFLOG\footnote{The NFLOG interface is a virtual network interface to receive the packets that have been logged by the kernel packet filter.} interface.
  %On the other hand, we could also capture network traffic with noise which included packets from other apps and operating systems if we sniffed the physical interface of the target device at the same time.
  We also filtered out retransmit, out of order and zero-payload packets.%, and then gathered the side-channel information from each network flow such as the byte size and arrived timestamp of packets in the flow.
\end{itemize}
We collected our private dataset from 23rd June and obtained a dataset named D1.
Then we collected dataset D2 one month later after the D1 was captured, as Table~\ref{datasets} depicts, and 22 apps have been updated.


