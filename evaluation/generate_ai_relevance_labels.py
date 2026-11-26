"""Generate AI-assisted labels from the project's movie metadata.

The judgments below are grounded in the actual dataset's overview, keywords,
genres, cast, and director fields. Similarity scores are never read. Missing or
insufficient story information is left unrated for human review.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evaluation"
DATA = ROOT / "data" / "processed" / "unified_movies_enriched.csv"
RECOMMENDATIONS = EVAL / "model_recommendations.csv"
RATING_SHEET = EVAL / "relevance_labels.csv"
AI_LABELS = EVAL / "ai_relevance_labels.csv"

# Each item is (rating, metadata-grounded reasoning, confidence). A None rating
# means the available metadata is insufficient and requires manual review.
ASSESSMENTS = {
    155: {
        "Batman": (3, "The overview places Batman against the Joker in Gotham; it directly shares the hero, city, and central conflict.", "High"),
        "Batman Begins": (3, "Bruce Wayne becomes Gotham's crime-fighting vigilante, closely matching the character and crime-focused premise.", "High"),
        "Batman Returns": (3, "A Batman-led Gotham crime story with recurring characters and villains is directly aligned.", "High"),
        "Batman Forever": (3, "Batman protects Gotham from criminal villains, directly matching the seed's hero and setting.", "High"),
        "Batman & Robin": (3, "The same Batman, Gotham setting, and superhero-versus-criminal conflict make this directly relevant.", "High"),
        "Hitman": (1, "Both are crime-action stories involving a masked or concealed identity, but the assassin premise differs substantially.", "Medium"),
        "Dark Blue": (2, "A corruption-focused homicide investigation and a morally compromised officer overlap with the seed's urban crime and justice themes.", "Medium"),
        "Watchmen": (3, "A dark, morally complicated story about masked vigilantes investigating violence strongly matches the superhero-noir concept.", "High"),
        "Law Abiding Citizen": (2, "Vigilante justice, crime, and a district attorney provide meaningful overlap, though the protagonist and plot differ.", "Medium"),
        "Kick-Ass": (2, "A self-made masked crime fighter and comic-book superhero premise overlap, but the tone and story are notably different.", "Medium"),
        "Defendor": (2, "An ordinary man adopts a vigilante identity to fight urban crime, a meaningful but not exact parallel.", "Medium"),
        "The Dark Knight Rises": (3, "A direct Batman/Gotham continuation with the same vigilante and crime-conflict framework.", "High"),
        "All Superheroes Must Die": (2, "Masked vigilantes face an archenemy in a superhero crime scenario, but the story is otherwise distinct.", "Medium"),
        "Batman: The Dark Knight Returns, Part 2": (3, "The Batman identity, Gotham, and a returning Joker conflict directly match the seed's concept.", "High"),
        "Batman v Superman: Dawn of Justice": (3, "Bruce Wayne's Gotham vigilante is a central character, with the same comic-book hero and justice themes.", "High"),
        "Amidst the Devil's Wings": (None, "Manual review required: the overview only says it is a prequel to another title; it does not describe the story or establish a connection to The Dark Knight.", "Low"),
    },
    1726: {
        "The Incredible Hulk": (2, "A Marvel superhero origin and a hero pursued for his powers offer meaningful shared superhero-universe themes, but no Iron Man storyline is described.", "Medium"),
        "Captain America: The First Avenger": (2, "A Marvel super-soldier origin and world-saving hero premise overlap with Iron Man's superhero concept, though the period and plot differ.", "Medium"),
        "Steel": (1, "A hero uses advanced metal armor to fight criminals, a weak concept-level match without shared story or universe.", "Medium"),
        "Iron Man 2": (3, "The overview explicitly continues Tony Stark's armored-hero story and its public consequences.", "High"),
        "The Avengers": (3, "Tony Stark/Iron Man joins a superhero team to confront a global threat, directly continuing the character and concept.", "High"),
        "X-Men": (2, "A superhero team and comic-book action offer meaningful genre and subject overlap, but the characters and conflict differ.", "Medium"),
        "Super": (2, "An ordinary person becomes a costumed crime-fighting hero, paralleling the superhero transformation but with a different story.", "Medium"),
        "Man of Steel": (2, "A technologically distinct hero discovers his powers and must save the world; the superhero-origin and responsibility themes overlap.", "Medium"),
        "X-Men: First Class": (2, "A team of powered heroes and a global threat create meaningful superhero overlap, without the Iron Man character or technology premise.", "Medium"),
        "Iron Man 3": (3, "Tony Stark and his Iron Man identity are central, making this a direct continuation.", "High"),
        "The Wolverine": (1, "A Marvel superhero confronts a personal nemesis, but the character and story have little direct connection to Iron Man.", "Medium"),
        "Avengers: Age of Ultron": (3, "Tony Stark's actions and the Avengers are central to a world-threatening conflict; it directly continues the seed's character and universe.", "High"),
        "Captain America: The Winter Soldier": (2, "The Avengers/SHIELD superhero world overlaps, but the story centers on Captain America rather than Iron Man.", "Medium"),
        "Ant-Man": (2, "A technologically enabled superhero and Marvel universe make a relevant concept-level match, with a different hero and plot.", "Medium"),
        "Guardians of the Galaxy": (2, "A team of Marvel heroes faces an alien threat, providing meaningful shared-universe and superhero themes.", "Medium"),
        "Captain America: Civil War": (3, "Iron Man/Tony Stark is a central protagonist in a conflict among superheroes, directly continuing the character's story.", "High"),
    },
    20453: {
        "About Schmidt": (1, "Both involve personal reflection and life choices, but this retired man's story has no student, college, or education connection.", "Medium"),
        "The Bucket List": (1, "Male friendship and life lessons are broad overlaps; the story is otherwise about terminal illness and end-of-life goals.", "Medium"),
        "Like Stars on Earth": (3, "A child struggling within a rigid education system and a teacher who recognizes his potential strongly match the seed's education and mentorship themes.", "High"),
        "Superbad": (1, "Teen friendship and approaching adulthood are weak coming-of-age similarities; the plot is a very different night-long comedy.", "Medium"),
        "The Sisterhood of the Traveling Pants": (1, "Friendship and growing up are broad common themes, but the story centers on four girls' summer experiences.", "Medium"),
        "Loser": (2, "A college student navigating university life, roommates, and relationships offers meaningful student-life overlap.", "Medium"),
        "Driving Lessons": (1, "A teenager's coming-of-age and relationship with an older mentor are weak similarities; the story is not about education or student life.", "Medium"),
        "The Best Man": (2, "The plot reunites a college friendship circle, making friendship and student-era relationships a meaningful overlap.", "Medium"),
        "The Motel": (1, "A young boy's coming-of-age and mentorship offer a weak personal-growth parallel, but not the college or education themes.", "Medium"),
        "Smoke Signals": (1, "The story follows two young men and friendship, but its family-history journey differs from the seed's campus and education focus.", "Medium"),
        "Aamdani Atthanni Kharcha Rupaiya": (1, "A broad comedy about household finances and several couples has little connection to college, students, or education.", "Medium"),
        "Funny Ha Ha": (2, "A young adult navigating life after college and uncertainty about the future meaningfully overlaps with the seed's post-student life themes.", "Medium"),
        "Xcuse Me": (2, "Young unemployed men enter a training program and scheme their way through it, providing a relevant education/youth comedy connection.", "Medium"),
        "Hum Hain Rahi Pyar Ke": (1, "Family comedy and young people are broad overlaps, but the plot focuses on children and a runaway bride rather than campus life.", "Medium"),
        "Dil": (1, "College rivalry and romance provide some student-life overlap, but the central story is a romantic conflict.", "Medium"),
        "There Goes My Baby": (2, "High-school friends confronting separation and future choices create a meaningful coming-of-age and friendship parallel.", "Medium"),
        "Jhoothi": (1, "A comedy involving lies and relationships has only broad tonal overlap, without the seed's student or educational subject.", "Medium"),
        "Tauba Tauba": (0, "The synopsis centers on sexual obsession and adult misconduct, not friendship, education, or the seed's coming-of-age themes.", "Medium"),
        "Chaarfutiya Chhokare": (2, "A woman building a school encounters vulnerable village boys; education and youth are meaningful shared subjects, though the plot is a social thriller.", "Medium"),
        "Dope": (2, "A high-school student balancing college applications and friendship provides meaningful youth, education, and student-life overlap.", "Medium"),
        "Naunihal": (2, "An orphan's school life and relationship with a principal offer a meaningful education and mentorship connection, though the story differs.", "Medium"),
        "Shararat": (None, "Manual review required: the available overview only says '1959 Bollywood Drama Comedy'; it provides no story details beyond broad genres.", "Low"),
        "Tu Hai Mera Sunday": (1, "A group of friends and their shared activities are a weak friendship parallel, but the story is about adults seeking a place to play football.", "Medium"),
        "Hindi Medium": (3, "The story directly examines the education system, school admissions, and family pressures, strongly matching the seed's education theme.", "High"),
        "Hichki": (3, "A teacher works against a flawed system to educate overlooked students, closely matching the seed's student and mentorship themes.", "High"),
        "Jhootha Kahin Ka": (2, "Two young men go abroad to study and resist returning, providing meaningful student-life comedy overlap.", "Medium"),
        "Jolly LLB 3": (0, "The synopsis concerns a widow's legal case and a judicial satire, with no meaningful student, friendship, or education connection.", "Medium"),
        "Madgaon Express": (1, "Three childhood friends on a chaotic trip create a weak friendship-comedy parallel but not the seed's education themes.", "Medium"),
        "4 Idiots": (3, "The dataset explicitly describes it as a sequel to 3 Idiots, establishing a direct narrative/franchise relationship.", "High"),
        "Jacob Cardoso": (None, "Manual review required: overview and keywords are absent; romance/drama/comedy genres alone cannot establish story relevance.", "Low"),
        "Aryabhatt Ka Zero": (2, "A directionless young man struggles to find his place and undertakes self-discovery, meaningfully echoing the seed's growth and life-direction themes.", "Medium"),
    },
    21614: {
        "Office Space": (2, "Three workers attempt to embezzle money from their employer; the comic group scheme and money motive meaningfully overlap.", "Medium"),
        "Half Baked": (1, "Friends plan to free someone from jail in a crime comedy, a weak ensemble-comedy/crime overlap with a different motive.", "Medium"),
        "Let's Go to Prison": (1, "A revenge-driven criminal sends a man's son to prison, sharing crime-comedy elements but not the seed's ransom-and-scheme plot.", "Medium"),
        "Analyze That": (2, "A gangster and his analyst create a crime-comedy built around criminal life and comic problems, a meaningful tone/subject overlap.", "Medium"),
        "Police Academy: Mission to Moscow": (1, "Police and mafia are treated comically, but the law-enforcement plot has little in common with the seed's mistaken ransom scheme.", "Medium"),
        "Welcome": (2, "Gangsters disrupt a respectable family's marriage plans in a comic chain of complications, meaningfully overlapping with crime comedy.", "Medium"),
        "Phir Hera Pheri": (3, "This is a direct sequel featuring the same characters, whose greed and financial scam lead to another comic money crisis.", "High"),
        "Malamaal Weekly": (3, "A group schemes around a lottery winner's money, strongly matching the seed's accidental windfall, greed, and comic con plot.", "High"),
        "Aamdani Atthanni Kharcha Rupaiya": (2, "Several households struggle with money and wages in a comedy, a meaningful financial-strain and ensemble-comedy connection.", "Medium"),
        "King's Ransom": (3, "A businessman stages his own kidnapping to protect his wealth, but other kidnappers intervene; the money-driven ransom caper strongly matches.", "High"),
        "An Everlasting Piece": (1, "Two partners make money selling wigs in a comedy, a weak hustling/partnership similarity without the seed's criminal ransom scheme.", "Medium"),
        "Kaante": (2, "An ensemble of criminals plans a bank robbery, meaningfully overlapping on group crime and money, though its tone and plot differ.", "Medium"),
        "Apna Sapna Money Money": (2, "A young conman uses disguises and schemes to make quick money, a meaningful comedy-and-deception parallel.", "Medium"),
        "Andaz Apna Apna": (1, "Two slackers and a criminal threat create a broad comic-crime overlap, but the romance/protection story is different.", "Medium"),
        "Small Apartments": (1, "A deadbeat hides an accidental killing in a black comedy, a weak crime-comedy similarity without the seed's money/ransom plot.", "Medium"),
        "Phir Bhi Dil Hai Hindustani": (1, "A murder case and reporters create a crime-comedy context, but its focus is proving innocence rather than comic money schemes.", "Medium"),
        "Identity Thief": (2, "An identity theft and road pursuit built around criminal deception provides meaningful crime-comedy and scam overlap.", "Medium"),
        "Kabhi Na Kabhi": (1, "The plot involves an underworld boss and criminal tasks, but lacks the seed's comic tone and ransom scheme.", "Medium"),
        "We're the Millers": (2, "A fake family transports drugs in a comic caper, meaningfully overlapping on improvised schemes, crime, and comedy.", "Medium"),
        "Haque": (1, "A woman seeks justice after an accident involving her politician husband; crime and justice are present, but not the seed's comedy or scheme.", "Medium"),
        "Bhayaanak Panjaa": (0, "The synopsis is a supernatural murder-reporting horror story, unrelated to the seed's comedy, money, or ransom plot.", "Medium"),
        "Ascharya Fuck It": (0, "The overview describes a sexually charged love story involving a star, chauffeur, prostitute, and pimp, with no meaningful seed connection.", "Medium"),
        "Jeevan Jyoti": (0, "The synopsis focuses on marriage and suspicion of infidelity, unrelated to the seed's comic crime scheme.", "Medium"),
        "Maa Behen": (2, "A mother and daughters cover up a crime in their neighborhood, meaningfully overlapping on family members caught in a comic/crime predicament.", "Medium"),
        "Jeevan Bheema Yojana": (3, "A debt-ridden couple devises an insurance scheme involving a lookalike; the comic fraud and money motive strongly match.", "High"),
        "Reporting Live": (1, "A reporter gets entangled in an illegal mining racket, sharing crime but not the seed's comedy or ransom/caper premise.", "Medium"),
        "Pratap": (0, "A police specialist imprisoned for murder and seeking self-discovery has no meaningful storyline or tonal connection.", "Medium"),
    },
    157336: {
        "20,000 Leagues Under the Sea": (1, "A mysterious voyage aboard an advanced submarine is an exploration adventure, but it is underwater rather than space travel.", "Medium"),
        "Solaris": (2, "A space-station mission encounters phenomena tied to human consciousness, making a meaningful high-concept space and existential-sci-fi parallel.", "Medium"),
        "A.I. Artificial Intelligence": (2, "Artificial intelligence, humanity, and a parent-child bond create meaningful thematic overlap, though the plot is not an interstellar mission.", "Medium"),
        "Silent Running": (2, "A spacecraft crew tries to preserve Earth's last ecosystems, sharing space travel, survival, and humanity-focused stakes.", "Medium"),
        "Close Encounters of the Third Kind": (2, "An unexplained extraterrestrial encounter and humanity's response provide meaningful science-fiction and cosmic-contact overlap.", "Medium"),
        "Galaxy Quest": (1, "A crew travels aboard an alien spacecraft, but the story is primarily a parody comedy rather than serious exploration or survival.", "Medium"),
        "Mission to Mars": (3, "A crew launches a rescue expedition to Mars after contact is lost, strongly matching space exploration and mission-survival concepts.", "High"),
        "Lost in Space": (2, "A family is launched into space to establish a colony, meaningfully matching interplanetary travel and human survival.", "Medium"),
        "The Astronaut's Wife": (1, "An astronaut returns from a near-fatal mission, but the story centers on his marriage and suspected transformation rather than exploration.", "Medium"),
        "The Astronaut Farmer": (2, "An astrophysicist builds a rocket to pursue spaceflight, echoing ambition and the practical challenges of space travel.", "Medium"),
        "Event Horizon": (2, "A crew investigates a vanished spacecraft near a black-hole-like phenomenon, sharing deep-space mission and black-hole concepts despite its horror focus.", "Medium"),
        "Timeline": (2, "Quantum mechanics and time travel drive a rescue expedition, offering meaningful scientific and time-related concept overlap.", "Medium"),
        "The Black Hole": (3, "A spacecraft crew encounters a lost ship near a black hole, strongly matching space exploration and the seed's black-hole concept.", "High"),
        "Meet Dave": (1, "Aliens operate a human-shaped spacecraft to save their planet, a broad space-science-fiction overlap presented as comedy.", "Medium"),
        "Stargate: The Ark of Truth": (2, "A team travels through wormholes to confront an intergalactic threat, meaningfully overlapping on wormholes and space missions.", "Medium"),
        "Gravity": (3, "Astronauts face a catastrophic accident and struggle to survive in space, strongly matching space-mission danger and survival.", "High"),
        "Prometheus": (3, "An expedition explores the origins of humanity in deep space and faces existential danger, closely matching the seed's exploration and human-survival themes.", "High"),
        "Hard to Be a God": (2, "Scientists are sent to another planet to influence a civilization, a meaningful but differently focused interplanetary expedition.", "Medium"),
        "About Time": (2, "Time travel and a central father-son relationship provide meaningful narrative-concept and family overlap, though the genre and plot differ.", "Medium"),
        "Space Pirate Captain Harlock": (2, "A spacecraft crew confronts an alien threat to Earth, offering a meaningful space-voyage and planetary-survival connection.", "Medium"),
        "Midnight Special": (2, "A father protects his extraordinary son while on the run, echoing the seed's parent-child bond and high-stakes science-fiction themes.", "Medium"),
        "Ex Machina": (1, "Artificial intelligence and human nature are shared broad science-fiction themes, but the contained story has no exploration or space-mission link.", "Medium"),
        "The Martian": (3, "An astronaut is stranded on Mars and uses science to survive until rescue, strongly matching exploration, isolation, and problem-solving.", "High"),
        "Dawn of the Crescent Moon": (0, "Its synopsis concerns students investigating a massacre legend; it provides no meaningful space, exploration, or cosmic-science connection.", "Medium"),
    },
    352173: {
        "Zodiac": (2, "An obsessive criminal investigation and evidence trail provide meaningful crime-mystery overlap, though the seed focuses on a family's concealment.", "Medium"),
        "Jindabyne": (2, "A group discovers a murdered woman and delays reporting it, creating a meaningful parallel around concealment, guilt, and a crime investigation.", "Medium"),
        "The Pelican Brief": (2, "A witness uncovers a murder conspiracy and investigates while in danger, meaningfully overlapping on hidden truth and criminal investigation.", "Medium"),
        "High Crimes": (2, "A lawyer discovers her husband has concealed his identity and faces a murder trial, creating meaningful deception and legal-investigation overlap.", "Medium"),
        "Freeze Frame": (3, "A murder suspect records himself to create an alibi, directly matching the seed's use of evidence and elaborate efforts to evade suspicion.", "High"),
        "The Betrayed": (1, "A mother is forced into a criminal recovery scheme because of her husband's actions, a weak family-and-crime parallel.", "Medium"),
        "The Truman Show": (2, "A man's entire life is secretly staged and recorded, meaningfully echoing hidden truth, surveillance, and a constructed cover story.", "Medium"),
        "Gone Girl": (3, "A disappearance, media suspicion, and a spouse's possible framing create a strong deception-and-investigation match.", "High"),
        "Black Water Transit": (1, "A business owner is caught amid criminal activity and corruption, but the synopsis offers little connection to family concealment or investigation.", "Medium"),
        "Ittefaq": (3, "Two witnesses to a double murder are also suspects in a police investigation, closely matching the seed's crime, suspects, and deception themes.", "High"),
        "Bhayaanak Panjaa": (1, "Reporters investigate a suspected murder at a haunted mansion, a weak crime/mystery overlap with a supernatural focus.", "Medium"),
        "Taish": (1, "A revenge conflict between violent families shares crime and family tension but not the seed's concealment or investigative strategy.", "Medium"),
        "Barot House": (2, "A family is torn apart by killings and a mystery about who is responsible, providing meaningful family-crime overlap.", "Medium"),
        "My Client's Wife": (3, "A lawyer investigates an alleged assault after the accused claims he is framed, closely matching the seed's false-evidence and truth-concealment themes.", "High"),
        "Jaane Jaan": (3, "A mother and daughter become involved in a deadly crime while a neighbor helps and a cop investigates, strongly matching family protection and concealment.", "High"),
        "Drishyam 2": (3, "It directly continues the Salgaonkar family case and the threat that truth will emerge, a direct sequel and storyline match.", "High"),
        "Baby John": (1, "A father protects his daughter from criminals and seeks revenge, a broad family-protection/crime overlap with a different action premise.", "Medium"),
        "Deva": (2, "A police officer must secretly reinvestigate a murder after memory loss, meaningfully overlapping on investigation and hidden information.", "Medium"),
        "Sector 36": (2, "A cop investigates disappearances linked to a serial killer, offering meaningful crime-investigation and suspense overlap.", "Medium"),
        "Drishyam: The Conclusion": (3, "The family's buried secret resurfaces and Vijay must outwit investigators again, a direct continuation of the seed's plot.", "High"),
        "The Confession of Mary": (2, "A murder mystery turns on concealed truth and the limits of confession, a meaningful but distinct crime-and-secrecy parallel.", "Medium"),
        "Reporting Live": (1, "A reporter and cameraman uncover an illegal racket, sharing crime investigation but not the seed's family cover-up.", "Medium"),
        "Raat Akeli Hai - The Bansal Murders": (2, "A family murder investigation exposes greed, betrayal, and secrets, meaningfully overlapping on family secrets and crime.", "Medium"),
        "Gajab Raat": (1, "A troubled cop becomes entangled in a criminal web, a weak police/crime overlap without the seed's concealment premise.", "Medium"),
        "Ikka": (2, "A lawyer must defend a man he suspects is guilty, creating a meaningful truth, guilt, and legal-ethics connection.", "Medium"),
        "Anjan": (1, "The synopsis mentions a household shaped by love and deceit, but offers little specific connection to the seed's crime concealment.", "Low"),
        "Bekhabar": (1, "A suspected gangster cannot be charged because of insufficient proof, a weak overlap on concealed evidence and police suspicion.", "Low"),
        "Babita Singh Reporting": (2, "A cop defies others to investigate her friend's murder, a meaningful crime-investigation and truth-seeking connection.", "Medium"),
        "Pratap": (1, "A police officer is imprisoned for murder and confronts a past tied to his arrest, a weak law-and-crime parallel.", "Medium"),
    },
    360814: {
        "Mother India": (1, "A mother's perseverance and moral struggle are broad family/inspirational parallels, but there is no sports or training storyline.", "Medium"),
        "My Girl": (1, "A child's family experiences and growing up are weak coming-of-age similarities, without the seed's sport or empowerment themes.", "Medium"),
        "Kit Kittredge: An American Girl": (2, "A determined young girl faces hardship during the Depression, meaningfully echoing the seed's resilient female-protagonist and overcoming-obstacles themes.", "Medium"),
        "Dear Frankie": (1, "A mother protects her child from an abusive father, a weak parent-child hardship parallel but not a sports or achievement story.", "Medium"),
        "Kabhi Khushi Kabhie Gham": (1, "Family estrangement and reunion are broad family-drama themes, but the seed's central sports and daughters' training are absent.", "Medium"),
        "The Longshots": (3, "A true story of an eleven-year-old girl breaking barriers in football strongly matches female athletic achievement and inspirational sports.", "High"),
        "Because of Winn-Dixie": (1, "A girl and her father form a new life in a small town, a weak family/growing-up overlap without sport or training.", "Medium"),
        "Hachi: A Dog's Tale": (1, "The story emphasizes loyalty and family bonds, only broadly related to the seed's parent-child inspirational themes.", "Medium"),
        "Marmaduke": (0, "The synopsis is a family comedy about a mischievous dog, with no meaningful sports, training, or perseverance connection.", "Medium"),
        "Stolen Summer": (1, "A boy faces religious and moral questions during childhood, a weak inspirational coming-of-age overlap.", "Medium"),
        "The Perfect Game": (3, "A true story of an underdog youth team winning a major baseball tournament strongly matches the seed's inspirational sports-biography concept.", "High"),
        "Damadamm!": (0, "A romantic comedy about a man and his relationships has no meaningful connection to sports, training, or female achievement.", "Medium"),
        "Annie": (1, "An orphaned girl overcomes hardship, a broad resilient-child parallel without the seed's sports or family-training story.", "Medium"),
        "Hawas": (0, "The synopsis concerns a man asked to seduce a gangster's daughter, unrelated to the seed's sports and empowerment themes.", "Medium"),
        "Joy": (2, "A struggling single mother becomes a successful inventor, meaningfully overlapping on determination and a woman's achievement, though not sports.", "Medium"),
        "Madhosh": (1, "A young man defies a conflict between two families for love, a weak family-struggle parallel without sport or training.", "Medium"),
        "Nauker": (0, "A wealthy widower swaps identities with a servant while seeking a bride, unrelated to the seed's core themes.", "Medium"),
        "Shivaay": (1, "A father undertakes a dangerous effort to protect his daughter, a weak parent-child perseverance parallel with a different action-adventure plot.", "Medium"),
        "Namkeen": (1, "A mother and her daughters face changing circumstances, a broad family-resilience parallel without a sports or achievement focus.", "Medium"),
        "M.S. Dhoni: The Untold Story": (3, "A true sports biography traces a cricketer's journey to championship success, strongly matching the seed's athlete-training and achievement story.", "High"),
        "Awesome Mausam": (0, "The synopsis is a romance between a Hindu boy and Muslim girl, with no meaningful connection to sport or the seed's central struggle.", "Medium"),
        "Goodbye": (1, "A family faces grief after a mother's death, a broad parent/family theme without sports or achievement.", "Medium"),
        "Rocky Aur Rani Kii Prem Kahaani": (1, "A couple from different families navigate cultural differences, a weak family relationship overlap rather than a sports story.", "Medium"),
        "Delhi Mellei": (1, "A young woman pursues education and faces setbacks, a weak female-determination parallel without sport or athletic training.", "Medium"),
        "Sunday": (1, "A father relocates to support his son's studies and faces personal strain, a weak parent-child and education connection.", "Medium"),
        "Lailaa Manju": (0, "The synopsis focuses on a romantic relationship and a mother's identity concerns, unrelated to sport or the seed's story.", "Medium"),
        "Breed": (1, "The story concerns a marginalized person's identity and social hardship, a broad resilience/empowerment theme but not sport.", "Low"),
        "Sumo Didi": (3, "A woman pursues sumo wrestling through rigorous training despite obstacles, directly matching the seed's female-athlete perseverance.", "High"),
        "Narmeen": (1, "A young mother grieves during Partition, a broad hardship story without the seed's sport or achievement themes.", "Medium"),
        "Daadi Ki Shaadi": (1, "A family wedding and social-media mishap create a family comedy, only a broad family-drama overlap.", "Medium"),
        "Baghban": (None, "Manual review required: the overview describes a different older family/prison story but appears too sparse and potentially unreliable to make a confident thematic comparison.", "Low"),
        "Danka": (1, "A person challenges injustice in a community, a broad perseverance theme but no sports, training, or family-athlete connection.", "Low"),
        "The Great Shamsuddin Family": (1, "An extended family faces a day of personal crises, a broad family-comedy connection without the seed's sports or achievement themes.", "Medium"),
        "Meethe Mein Kya Hai": (1, "A grieving family reunites and faces a marriage-related upheaval, a broad family theme but no sports or training.", "Medium"),
    },
    441889: {
        "D.E.B.S.": (0, "The synopsis is a teen-crime-fighter romance, unrelated to a young singer facing domestic abuse and pursuing music.", "Medium"),
        "Stomp the Yard": (2, "A young dancer develops his performance talent at university, meaningfully overlapping on artistic ambition and music/performance.", "Medium"),
        "Once": (3, "Two aspiring singer-songwriters connect through music while pursuing a break, strongly matching the seed's musical talent and artistic ambition.", "High"),
        "Divine Secrets of the Ya-Ya Sisterhood": (1, "A mother-daughter dispute is resolved by family friends, a weak parent-child relationship parallel without music or an abusive-singer story.", "Medium"),
        "The Doors": (3, "A biographical music story follows a famous singer and band, strongly matching the seed's music-centered life and performance concept.", "High"),
        "Sherrybaby": (2, "A mother tries to reclaim her daughter after addiction and prison, meaningfully overlapping on domestic abuse/family hardship though music is absent.", "Medium"),
        "What's Love Got to Do with It": (3, "A female singer rises to fame while escaping an abusive relationship, an especially close match in music, abuse, and independence.", "High"),
        "La Bamba": (3, "A music biography follows a singer's rise from poverty to success, strongly matching the seed's young artist and musical career.", "High"),
        "Coal Miner's Daughter": (3, "A woman rises from poverty to become a celebrated singer, strongly matching the seed's female musical-ambition story.", "High"),
        "Disco Dancer": (3, "A performer rises from street singing to stardom despite people from his past, strongly matching the seed's music and career trajectory.", "High"),
        "Jashnn: The Music Within": (None, "Manual review required: the overview only repeats the title and keywords are absent; music/drama/romance genres do not establish its story.", "Low"),
        "Gupt: The Hidden Truth": (0, "The synopsis is a murder mystery about a man proving his innocence, with no meaningful music or singer connection.", "Medium"),
        "Masked and Anonymous": (2, "A benefit concert is organized amid unrest, providing a meaningful music/performance setting but little overlap with the seed's personal story.", "Medium"),
        "The Runaways": (3, "Two teenage girls become the frontwomen of a rock band, closely matching a young female artist's music career and ambition.", "High"),
        "Awaara": (1, "The synopsis follows a thief's murder trial and intertwined lives; music is only a genre label, so overlap is weak.", "Low"),
        "Eve's Bayou": (1, "A family drama involves a daughter and troubled parents, a weak family/abuse-adjacent theme but not a singer's story.", "Medium"),
        "To Write Love on Her Arms": (1, "The story centers on a young woman's addiction and depression; those are weak personal-hardship parallels, not a music-career match.", "Medium"),
        "Sparkle": (3, "A musical prodigy struggles to become a star amid family problems, strongly matching the seed's singer and family-conflict themes.", "High"),
        "Ek Chhoti Si Love Story": (0, "The overview describes an adult's sexual relationship with a teenager, unrelated to the seed's music and self-expression story.", "Medium"),
        "Aashiqui 2": (3, "A failing musician launches a young woman's singing career as their relationship is tested by fame, closely matching music and romance themes.", "High"),
        "Grace Unplugged": (3, "A young singer-songwriter pursues stardom while navigating family ties, directly matching artistic ambition and family conflict.", "High"),
        "Arjun Pandit": (0, "The synopsis is about betrayal, gangster violence, and forced marriage, unrelated to music or a young singer's journey.", "Medium"),
        "Straight Outta Compton": (3, "A music group uses songs to express life in a difficult environment and rises to fame, strongly matching the seed's music and aspiration.", "High"),
        "Mardaani": (0, "A police officer investigates child trafficking, unrelated to the seed's music, identity, or singer-ambition themes.", "Medium"),
        "Bajrangi Bhaijaan": (1, "A man helps reunite a lost child with her family, a weak compassion/family connection without a music or singer storyline.", "Medium"),
        "Mom": (1, "A mother seeks justice after her stepdaughter is assaulted, a broad family-protection/abuse theme but not a music-career story.", "Medium"),
        "Pistolwali": (0, "The synopsis is a revenge story about a woman trained to avenge her mother, with no meaningful music connection.", "Medium"),
        "Gully Boy": (3, "An underdog from Mumbai's Dharavi finds his voice and develops as a rapper, directly matching the seed's aspiring singer and self-expression themes.", "High"),
        "Hacked": (0, "A thriller about a rejected hacker's obsession has no meaningful music, performance, or family-story overlap.", "Medium"),
        "Madam Fashion": (1, "The overview describes a modern woman and glamour, but does not establish a singer or music story; overlap is weak.", "Low"),
        "Starfruits": (1, "A separated couple and search for identity are broad personal themes, but the synopsis does not indicate music or performance.", "Low"),
        "Saiyaara": (2, "Two artists connect through music and face challenges to their relationship, a meaningful music/romance connection but not the seed's abuse-to-stardom story.", "Medium"),
        "Gajab Raat": (0, "A corrupt cop becomes entangled in crime, unrelated to the seed's singer, music, and family themes.", "Medium"),
        "Tu Hai Meri Kiran": (0, "The synopsis is an obsessive romance thriller, with no meaningful music or aspiring-artist connection.", "Medium"),
    },
}


def main() -> None:
    recommendations = pd.read_csv(RECOMMENDATIONS, encoding="utf-8-sig")
    labels = pd.read_csv(RATING_SHEET, encoding="utf-8-sig")
    movies = pd.read_csv(DATA, encoding="utf-8-sig").fillna("")

    unique_pairs = recommendations[
        ["seed_movie", "seed_movie_id", "recommended_movie", "recommended_movie_id"]
    ].drop_duplicates()
    if unique_pairs.duplicated(["seed_movie_id", "recommended_movie_id"]).any():
        raise ValueError("A seed/candidate ID pair has conflicting title records.")

    data_by_id = movies.set_index("tmdb_id", drop=False)
    rows = []
    rating_by_pair = {}
    for pair in unique_pairs.itertuples(index=False):
        seed_id = int(pair.seed_movie_id)
        candidate_id = int(pair.recommended_movie_id)
        if seed_id not in data_by_id.index or candidate_id not in data_by_id.index:
            raise ValueError(f"Dataset metadata missing for pair {seed_id}/{candidate_id}.")
        seed = data_by_id.loc[seed_id]
        candidate = data_by_id.loc[candidate_id]
        if seed["title"] != pair.seed_movie or candidate["title"] != pair.recommended_movie:
            raise ValueError(
                f"Title/ID mismatch for pair {seed_id}/{candidate_id}: "
                f"{pair.seed_movie!r}/{seed['title']!r}, "
                f"{pair.recommended_movie!r}/{candidate['title']!r}."
            )
        try:
            rating, reasoning, confidence = ASSESSMENTS[seed_id][pair.recommended_movie]
        except KeyError as exc:
            raise ValueError(
                f"No AI assessment for pair {seed_id}/{candidate_id} "
                f"({pair.seed_movie} -> {pair.recommended_movie})."
            ) from exc
        review_status = "AI-rated" if rating is not None else "Manual review required"
        rows.append({
            "seed_movie": pair.seed_movie,
            "seed_movie_id": seed_id,
            "recommended_movie": pair.recommended_movie,
            "recommended_movie_id": candidate_id,
            "relevance_rating": "" if rating is None else rating,
            "reasoning": reasoning,
            "confidence": confidence,
            "evaluation_method": "AI-assisted",
            "review_status": review_status,
        })
        rating_by_pair[(seed_id, candidate_id)] = "" if rating is None else rating

    if len(rows) != len(unique_pairs):
        raise ValueError(f"Expected {len(unique_pairs)} pair judgments; generated {len(rows)}.")

    ai_labels = pd.DataFrame(rows).sort_values(
        ["seed_movie_id", "recommended_movie_id"]
    )
    ai_labels.to_csv(AI_LABELS, index=False, encoding="utf-8-sig")

    labels["relevance_rating"] = [
        rating_by_pair[(int(row.seed_movie_id), int(row.recommended_movie_id))]
        for row in labels.itertuples(index=False)
    ]
    reasoning_by_pair = {
        (int(row["seed_movie_id"]), int(row["recommended_movie_id"])): row["reasoning"]
        for row in rows
    }
    labels["reviewer_notes"] = [
        "AI-assisted: " + reasoning_by_pair[(int(row.seed_movie_id), int(row.recommended_movie_id))]
        for row in labels.itertuples(index=False)
    ]
    labels.to_csv(RATING_SHEET, index=False, encoding="utf-8-sig")

    manual = ai_labels[ai_labels["relevance_rating"] == ""]
    print(f"Unique pairs evaluated: {len(ai_labels)}")
    print(f"Repeated recommendation rows avoided: {len(recommendations) - len(unique_pairs)}")
    print(f"AI-rated pairs: {len(ai_labels) - len(manual)}")
    print(f"Pairs requiring manual review: {len(manual)}")
    print("Confidence distribution:")
    print(ai_labels["confidence"].value_counts().sort_index().to_string())
    print(f"Saved {AI_LABELS}")
    print(f"Updated {RATING_SHEET} (existing columns preserved)")


if __name__ == "__main__":
    main()
