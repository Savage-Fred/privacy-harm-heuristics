# Developing Explainable Heuristics, Models, and Tools for Privacy Risk Analysis Using Real-world Data 

> Practicum artifact (Dec 2025), extracted from the privacy-heuristics repo 2026-07-21.

## ***Name:*** Will McCarty

## **Created:** Aug 23, 2025

## **Updated:** Sep 13, 2025

## **Version:** 2 

## **Context:** Georgia Tech PUBP 6727 PrivacyEngine Project Proposal 

## **Email:** wmccarty6@gatech.edu

***TL;DR:** This project will build a dataset of real-world privacy harms and use interpretable models (e.g., Bayesian Rule Lists) to develop practical, explainable heuristics that bridge the gap between abstract privacy theory and empirical reality. Given time, budget, and computational constraints, the deliverables are aspirational and lay out a path for collaboration with Georgia Tech faculty and fellow privacy researchers.* 

# 1\. Challenge and Problem Statement

Existing privacy frameworks—such as [Solove’s taxonomy of privacy harms](https://scholarship.law.upenn.edu/penn_law_review/vol154/iss3/1/) (2006), [Nissenbaum’s theory of contextual integrity](https://digitalcommons.law.uw.edu/wlr/vol79/iss1/10/) (2004), and Westin’s [segmentation of privacy attitudes](https://scholarlycommons.law.wlu.edu/wlulr/vol25/iss1/20/#:~:text=Recommended%20Citation&text=Westin%2C%20Privacy%20And%20Freedom%2C%2025,166%20\(1968\).) (1968)—are conceptually influential but rarely validated against real-world evidence. They offer limited predictive value for practitioners. For instance, while contextual integrity emphasizes the importance of context, empirical studies show that users’ privacy judgments depend on specific combinations of factors such as data type, recipient, and purpose, not broad abstractions (Martin & Shilton, 2016; Martin & Nissenbaum, 2016). Similarly, Solove’s taxonomy has been [applied retrospectively](https://techscience.org/a/2018100903/#Citation) to a small dataset of 44 examples to categorize privacy incidents (Garfinkel & Theofanos, 2018), but the authors provide little guidance on which harms are most likely to occur in practice and the small dataset lacks the desirable effect of *Law of Large Numbers[^1]* ([Yao, Gao](https://ui.adsabs.harvard.edu/abs/2016ITFS...24..615Y/abstract)) applicability that would lead to generalizable advice. 

Practitioners today often rely on heuristics, ad hoc checklists, or compliance frameworks such as the [NIST Privacy Framework](https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.01162020.pdf) (NIST, 2020), but these tools lack empirical grounding in user reactions or documented harms. Recent research has [highlighted this gap](https://doi.org/10.4018/IJISP.2020100105): user reviews reveal widespread privacy distrust (Besmer et al., 2020), [social media analyses](https://doi.org/10.1089/cyber.2021.0188) show spikes of outrage and skepticism after major incidents (Lee et al., 2022), and [regulatory analyses of GDPR fines](https://www.researchgate.net/publication/361208074_Investigating_GDPR_Fines_in_the_Light_of_Data_Flows) emphasize recurring failures in consent and lawful basis (Saemann et al., 2022). Yet these insights remain siloed and descriptive, not operationalized into predictive or explainable tools. Efforts in adjacent domains—such as [interpretable risk models in medicine](https://arxiv.org/abs/1511.01644%20) (Letham et al., 2015\) — show the potential of combining real-world evidence with interpretable models. Others have identified similar problems such as Freiberger and Fleig’s [explainable AI-based policy analyzers](https://arxiv.org/html/2504.12931v1) proposal that “showcase(s) the application area of usable privacy and security to be promising for Human-Centered Explainable AI (HCXAI) to make an impact” (2025). 

With all of this as background, the specific problem I address is this: there is no empirically grounded, explainable framework and associated automated tools that translate real-world privacy harms and user reactions into actionable heuristics for practitioners. My project aims to bridge this gap by compiling data from documented privacy incidents and user complaints and using interpretable models to generate practical, human-readable heuristics for privacy risk analysis with the downstream potential of automating the application of the framework using AI and to identify risk early in the software development lifecycle where mitigations can be applied most effectively. 

## Proposed Solution: 

To address this problem, I will develop a privacy risk analysis framework that is explicitly grounded in real-world evidence and structured for explainability. Rather than relying solely on abstract principles or compliance checklists, I will compile and analyze data from multiple sources of privacy harm:

* User sentiment and complaints from public forums and app reviews, where individuals explicitly describe features or practices as invasive, “creepy,” or untrustworthy.  
* Documented incidents and enforcement records such as FTC actions, GDPR rulings, and HIPAA breach reports, which provide objective evidence of harm and, in many cases, attach monetary penalties.  
* Regulatory and organizational reports that detail recurring failures in consent, transparency, and lawful basis, which are strong indicators of systemic risk.

From this evidence base, I will extract recurring risk signals—such as always-on sensors, opaque data sharing, or lack of user controls—and encode them into interpretable models (e.g., Bayesian Rule Lists, decision trees). These models are deliberately chosen for their ability to produce human-readable heuristics, such as “If biometric data is collected without clear user consent, then risk is high.”

The immediate deliverable will be a prototype framework and dataset that translates empirical findings into actionable heuristics for privacy practitioners. My long term goal is to create a tool that can be applied in the early stages of the software development lifecycle, when interventions are most cost-effective. Over time, this framework could be extended into semi-automated tools—leveraging AI to apply the heuristics at scale while still maintaining the explainability and human-in-the-loop oversight required for responsible privacy engineering. 

Because privacy is contextual and culturally dependent (Garfield, 2017), a  long term approach would automatically update heuristics based on user location and new data as privacy expectations evolve. This beyond the scope of this PrivacyEngine. 

# 2\. Motivation, Relevance, and Inspiration 

Privacy failures cause measurable harm: [66% of U.S. consumers say they would not trust a company after a breach](https://vercara.digicert.com/news/vercara-research-75-of-u-s-consumers-would-stop-purchasing-from-a-brand-if-it-suffered-a-cyber-incident), and over [80% of victims stop doing business with the offender](https://secureframe.com/blog/data-privacy-statistics). But harms extend beyond breaches. Subjective reactions—discomfort, distrust, perceptions of “creepiness”—erode adoption and brand value, yet existing theories rarely predict them. The well-documented privacy paradox further underscores the need for empirically grounded heuristics that reflect behavior, not just principles.

## Cross-Disciplinary Inspiration: 

To design the heuristics and tool, I will draw inspiration from risk analysis approaches in other fields:

* **Medicine:** Doctors use diagnostic questionnaires and checklists (like depression inventories or surgical safety checklists) to systematically narrow down issues – this project seeks an analogous “privacy risk questionnaire” for new tech products, grounded in data.  
* **Aviation:** Every incident or near-miss is logged and analyzed, and pre-flight checklists are designed from decades of empirical data to prevent known failure patterns. Similarly, I intend to create a kind of pre-launch privacy checklist derived from patterns observed in past privacy incidents.  
* **Cybersecurity:** Frameworks like threat modeling and quantitative risk scoring (FAIR, NIST) use data of past attacks to prioritize risks ￼. My approach aligns with this, but focused on privacy-specific threats (including those that are intangible, like reputation damage).

By learning from these disciplines, the project will ensure the heuristics are not just data-driven, but also presented in a usable format for practitioners (e.g. a simple rule list or scorecard as opposed to a complex algorithm).

## Business Case for a new Privacy Framework 

While compliance with regulations such as GDPR, CCPA, and the NIST Privacy Framework is necessary, organizations increasingly recognize that privacy is also a business differentiator. Numerous studies and real-world incidents show that consumer trust is tightly coupled to how companies handle personal data.

### Trust and Retention: 

Cisco’s 2021 Consumer Privacy Survey found that 76% of consumers say they will not buy from a company they do not trust with their data, and 32% actively switched providers over privacy concerns (Cisco, 2021). A later benchmark study demonstrated that companies with mature privacy practices experienced shorter sales delays and fewer costly breaches, highlighting privacy’s role as a growth enabler, not just a compliance cost (Cisco, 2022).

### Reputation and Market Impact: 

Public backlash following Facebook’s Cambridge Analytica scandal and WhatsApp’s 2021 terms update shows how quickly perceived overreach can erode adoption. WhatsApp lost millions of users to competitors such as Signal and Telegram after a single poorly explained change (Hern, 2021). Similarly, Target’s predictive analytics program caused reputational harm after it revealed a teenager’s pregnancy to her family, demonstrating that privacy harms extend beyond breaches to include “creepiness” and loss of dignity (Wagstaff, 2012).

### Quantifiable Risk: 

Vercara’s 2023 research reported that 75% of U.S. consumers would stop purchasing from a brand after a cyber incident, with long-term impacts on customer lifetime value and brand equity (Kringel, 2023). Privacy-related churn thus represents not only a reputational risk but a material financial risk.

### Beyond Compliance: 

Scholars such as Solove and Citron (2022) emphasize that legal compliance frameworks are often insufficient in capturing the full spectrum of privacy harms—particularly subjective harms like embarrassment, discomfort, or chilling effects. Nissenbaum’s (2004) theory of contextual integrity underscores that violating social norms around information flow can be just as damaging as violating laws. Zuboff (2019) further argues that surveillance-based business models can undermine long-term legitimacy and trust, creating systemic risks to both companies and societies.

Taken together, these findings demonstrate that privacy maturity drives trust, trust drives adoption, and adoption drives revenue. For product development teams, the implication is clear: privacy risk management is not only about regulatory avoidance but about ensuring long-term business viability. A framework that operationalizes consumer sentiment and documented harms into practical heuristics will allow leadership to make informed tradeoffs early in design, avoiding the costly and unpredictable fallout of user distrust.

# 3\. Methodology

## Data Collection: 

The first step will be building a large dataset of privacy “harm” examples. This will involve scraping or gathering information from multiple channels:

| Data Source | About |
| ----- | ----- |
| **Public Forums** | User complaints about privacy violations by scraping public platforms like Reddit and Twitter, using keyword filtering and sentiment analysis to identify relevant posts. |
| **Official Reports** | Objective privacy incidents by collecting data from official sources such as data breach chronologies, HIPAA breach portals, and FTC enforcement summaries. |
| **Other Data** | Use privacy-focused surveys, academic compilations of data harms, and specialized complaint websites to enrich the data.  |

***Table 3.1 –** Data Collection in General*

***Note:** A full list of Data Sources is available here: [PrivacyEngine Data Sources](https://docs.google.com/spreadsheets/d/1vwN6NoN2q-NOD8ye80U_nkJ6D9LGnk6LXkV6GKtomY4/edit?gid=0#gid=0)*

### Potential Roadblocks

AI companies such as OpenAI, Google, Anthropic, etc. have mined the internet for content and content creators are pushing back. More sites than ever have explicit terms and conditions related to scraping. If a site requires payment or forbids scraping, it will not be included. 

My expectation is that only government records and academic datasets will be useful for this course. 

## Data Processing:

I will process both unstructured user comments and structured breach reports to extract key features, such as the technology or practice involved, the sentiment or harm expressed, and the consequences (e.g., fines, number of users affected). From this, I will distill a taxonomy of recurring privacy risk factors—like “collects audio,” “unexpected data use,” “lacks transparency,” or “third-party sharing”—to support model development.

## Modeling and Analysis: 

Once the data is prepared, I will apply explainable simple and explainable machine learning models to find patterns and develop predictive heuristics. Crucially, I am not aiming for a black-box AI that magically predicts privacy issues – this would be inappropriate, in my view, for a human-in-the-loop field like privacy. Instead, I will experiment with interpretable, rules-based or transparent models. 

### Candidates Models (in Order of Preference)

| Model Type | Description | Example/Benefit |
| ----- | ----- | ----- |
| Bayesian Rule Lists (BRLs);  Decision Trees | Applied to create interpretable and accurate clinical risk models in medicine. Yield human-readable if-then rules. | [Stroke risk prediction](https://arxiv.org/abs/1511.01644): mirrors CHADS₂ score simplicity but with improved accuracy. "IF a product has always-on sensors AND no user controls, THEN privacy risk \= High," directly informing privacy review checklists. |
| Explainable Boosting Machines (EBMs) | A type of glass-box model that can show the weight of each feature in a prediction. Maintain high accuracy while remaining interpretable. | Reveals feature contribution to risk score (e.g., biometric data collection contributes "+0.8" to a risk score). |
| Bayesian Networks;  Sparse Linear Models | Can capture probabilistic relationships between factors. Allow incorporating expert knowledge priors. | Illustrates relationships like "use of personal data in a new context" combined with "lack of user consent" leading to high probability of user backlash. Blends empirical findings with existing theories. |

***Table 3.2 –** Candidate Models*

### Validation 

I plan to validate the model’s outputs against known case studies – for instance, would the rules identified have predicted the user anger over Facebook’s Cambridge Analytica scandal, or Google Glass’s failure due to privacy concerns? If the model flags similar factors (e.g. “unexpected data use without consent” in the Cambridge Analytica case), that’s a good sign. If it misses them or flags irrelevant factors, I’ll adjust the approach.

# 4\. Feasibility & Scope

Recognizing the limited timeframe of a semester, the project will be scoped to prioritize the most impactful pieces. 

**Building the dataset** of freely available privacy harms is a primary milestone – even if the predictive model remains basic, the dataset itself will be a valuable contribution (since, to my knowledge, no large, consolidated “privacy harm” dataset exists publicly). 

* Web scraping will be approached carefully: modern websites often have anti-scraping measures, so I will start with sources that allow API access or exports (for example, Pushshift for Reddit data or Twitter’s academic API). 

* If extensive scraping proves unfeasible, I will pivot to using existing collections of user comments (some research papers provide datasets of annotated tweets/posts about privacy) or focus on fewer, high-quality sources. 

**Computing power** is a concern

* Sentiment or NLP analysis on thousands of comments can be expensive – so I may use sampled data or lighter text analysis techniques if needed. 

* The goal is to demonstrate the concept rather than achieve Big Data scale. All data work will consider ethics: no personal user information beyond public comments will be collected, and I will respect terms of service for any platform used.

* I’m willing to exhaust my free-tier GCP and AWS compute, but will pay out of my own pocket up to a limit. Ideally (though unlikely) processing will be feasible on my laptop. 

# 5\. Expected Deliverables

By the end of the course, I aim to produce the following deliverables (in no particular order).

1. **Literature Review**: A synthesis of existing privacy frameworks highlighting their limitations against empirical data.

2. **Privacy Harm Dataset**:A structured collection of real-world privacy incidents and user complaints, with extracted features such as technology, context, and consequences.

3. **Prototype Risk Tool**: A simple, explainable model (e.g., Bayesian Rule List or decision tree) that flags potential privacy risks from product features or practices.

4. **Heuristics & Guidelines:** A set of actionable “risk signals” distilled from the data and models, designed as checklists or rules for practitioners.

5. **Final Report & Presentation**: A summary of methods, findings, and next steps, including limitations and opportunities for further research or collaboration.

# 6\. Conclusion and Outlook

This project will be successful if it either demonstrates gaps in current frameworks, validates them with data, or creates a novel dataset for future research. Even partial success will lay groundwork for grant proposals and collaboration. The focus is on quality of insights rather than scale, with ethical and feasible data practices guiding all collection and analysis.

# 7\. References

###### Besmer, A. R., Watson, J., & Banks, M. S. (2020). Investigating user perceptions of mobile app privacy: An analysis of user-submitted app reviews. International Journal of Information Security and Privacy, 14(4), 74–91. [https://doi.org/10.4018/IJISP.2020100105](https://doi.org/10.4018/IJISP.2020100105) 

###### Bonnie, E., & Fitzgerald, A. (2025). 110+ data privacy statistics: The facts you need to know in 2025\. SecureFrame. [https://secureframe.com/blog/data-privacy-statistics](https://secureframe.com/blog/data-privacy-statistics) 

###### Christian, B. (2020). The alignment problem. W. W. Norton & Company. [https://brianchristian.org/the-alignment-problem/](https://brianchristian.org/the-alignment-problem/) 

###### Cisco. (2021). 2021 consumer privacy survey. Cisco Systems. [https://www.cisco.com/c/en/us/about/trust-center/privacy/consumer-privacy-survey.html](https://www.cisco.com/c/en/us/about/trust-center/privacy/consumer-privacy-survey.html) 

###### Cisco. (2022). 2022 data privacy benchmark study. Cisco Systems. [https://www.cisco.com/c/en/us/about/trust-center/privacy/benchmark-study.html](https://www.cisco.com/c/en/us/about/trust-center/privacy/benchmark-study.html) 

###### Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating noise to sensitivity in private data analysis. In S. Halevi & T. Rabin (Eds.), Theory of cryptography conference (TCC 2006\) (pp. 265–284). Springer. [https://www.microsoft.com/en-us/research/publication/calibrating-noise-to-sensitivity-in-private-data-analysis/](https://www.microsoft.com/en-us/research/publication/calibrating-noise-to-sensitivity-in-private-data-analysis/) 

###### Freiberger, V., & Fleig, A. (2025). Explainable AI in usable privacy and security: Challenges and opportunities. arXiv. [https://arxiv.org/abs/2504.12931](https://arxiv.org/abs/2504.12931) 

###### Garfield, B. (2017). Privacy as a cultural phenomenon. Journal of Media Critiques, 3(10), 117–204. [https://doi.org/10.17349/Jmc117204](https://doi.org/10.17349/Jmc117204) 

###### Garfinkel, S., & Theofanos, M. (2018). Non-breach privacy events. Technology Science. [https://techscience.org/a/2018100903/](https://techscience.org/a/2018100903/) 

###### General Data Protection Regulation, Regulation (EU) 2016/679, 2016 O.J. (L 119). [https://eur-lex.europa.eu/eli/reg/2016/679/oj](https://eur-lex.europa.eu/eli/reg/2016/679/oj) 

###### Hern, A. (2021, January 24). WhatsApp loses millions of users after terms update. The Guardian. [https://www.theguardian.com/technology/2021/jan/24/whatsapp-loses-millions-of-users-after-terms-update](https://www.theguardian.com/technology/2021/jan/24/whatsapp-loses-millions-of-users-after-terms-update) 

###### Kringel, D. (2023). Vercara research: 75% of U.S. consumers would stop purchasing from a brand if it suffered a cyber incident. Vercara. [https://vercara.digicert.com/news/vercara-research-75-of-u-s-consumers-would-stop-purchasing-from-a-brand-if-it-suffered-a-cyber-incident](https://vercara.digicert.com/news/vercara-research-75-of-u-s-consumers-would-stop-purchasing-from-a-brand-if-it-suffered-a-cyber-incident) 

###### Lee, D. S., Jiang, T., Crocker, J., & Way, B. M. (2022). Social media use and its link to physical health indicators. Cyberpsychology, Behavior, and Social Networking, 25(2), 87–93. [https://doi.org/10.1089/cyber.2021.0188](https://doi.org/10.1089/cyber.2021.0188) 

###### Letham, B., Rudin, C., McCormick, T., & Madigan, D. (2015). Interpretable classifiers using rules and Bayesian analysis: Building a better stroke prediction model. Annals of Applied Statistics, 9(3), 1350–1371. [https://arxiv.org/abs/1511.01644](https://arxiv.org/abs/1511.01644) 

###### Lutz, C., & Newlands, G. (2021). Privacy and smart speakers: A multi-dimensional approach. The Information Society, 37(3), 147–162. [https://doi.org/10.1080/01972243.2021.1897914](https://doi.org/10.1080/01972243.2021.1897914) 

###### Martin, K., & Nissenbaum, H. (2017). Privacy interests in public records: An empirical investigation. Washington & Lee Law Review. [https://jolt.law.harvard.edu/articles/pdf/v31/31HarvJLTech111.pdf](https://jolt.law.harvard.edu/articles/pdf/v31/31HarvJLTech111.pdf) 

###### Martin, K., & Shilton, K. (2015). Experience, trust, and privacy in mobile space. Journal of the Association for Information Science and Technology, 66(12), 2705–2719. [https://doi.org/10.1002/asi.23340](https://doi.org/10.1002/asi.23340) 

###### National Institute of Standards and Technology. (2020). NIST privacy framework: A tool for improving privacy through enterprise risk management, version 1.0. U.S. Department of Commerce. [https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.01162020.pdf](https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.01162020.pdf) 

###### National Institute of Standards and Technology. (2024). NIST privacy framework 1.1 initial public draft. U.S. Department of Commerce. [https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.40.ipd.pdf](https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.40.ipd.pdf) 

###### Nissenbaum, H. (2004). Privacy as contextual integrity. Washington Law Review, 79(1), 119–158. [https://digitalcommons.law.uw.edu/wlr/vol79/iss1/10](https://digitalcommons.law.uw.edu/wlr/vol79/iss1/10) 

###### Organisation for Economic Co-operation and Development. (2013). OECD guidelines on the protection of privacy and transborder flows of personal data. OECD Publishing. [https://www.oecd.org/sti/ieconomy/privacy-guidelines.htm](https://www.oecd.org/sti/ieconomy/privacy-guidelines.htm) 

###### Ponemon Institute. (2018). Facebook privacy perceptions after Cambridge Analytica. Ponemon Research Brief.

###### Sämann, M., Theis, D., Urban, T., & Degeling, M. (2022). Investigating GDPR fines in the light of data flows. Proceedings on Privacy Enhancing Technologies, 2022(4), 64–84. [https://doi.org/10.56553/popets-2022-0095](https://doi.org/10.56553/popets-2022-0095) 

###### Solove, D. J. (2006). A taxonomy of privacy. University of Pennsylvania Law Review, 154(3), 477–564. [https://scholarship.law.upenn.edu/penn\_law\_review/vol154/iss3/1](https://scholarship.law.upenn.edu/penn_law_review/vol154/iss3/1) 

###### Solove, D. J., & Citron, D. K. (2022). Privacy harms. Boston University Law Review, 102(3), 793–850. [https://www.bu.edu/bulawreview/files/2022/07/SOLOVE-CITRON.pdf](https://www.bu.edu/bulawreview/files/2022/07/SOLOVE-CITRON.pdf) 

###### Wagstaff, K. (2012, February 17). How Target figured out a teen girl was pregnant before her father did. TIME. [https://techland.time.com/2012/02/17/how-target-figured-out-a-teen-girl-was-pregnant-before-her-father-did/](https://techland.time.com/2012/02/17/how-target-figured-out-a-teen-girl-was-pregnant-before-her-father-did/) 

###### Westin, A. F. (1968). Privacy and freedom. Atheneum.

###### Yao, K., & Gao, J. (2016). Law of large numbers for uncertain random variables. IEEE Transactions on Fuzzy Systems, 24(3), 615–621. [https://doi.org/10.1109/TFUZZ.2015.2466080](https://doi.org/10.1109/TFUZZ.2015.2466080) 

###### Zuboff, S. (2019). The age of surveillance capitalism: The fight for a human future at the new frontier of power. PublicAffairs.

[^1]:  Citation chosen as a modern application of the Law of Large numbers rather than citing the original proofs by Bernoulli and Poisson. 