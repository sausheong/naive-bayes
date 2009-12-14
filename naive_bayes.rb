require 'rubygems'
require 'stemmer'

class NaiveBayes

  # provide a list of categories for this classifier
  def initialize(categories)
    # keeps a hash of word count for each category
    @words = Hash.new		
    @total_words = 0		
    # keeps a hash of number of documents trained for each category
    @categories_documents = Hash.new		
    @total_documents = 0
    @threshold = 1.5
    
    # keeps a hash of number of number of words in each category
    @categories_words = Hash.new
    
    categories.each { |category|         
      @words[category] = Hash.new         
      @categories_documents[category] = 0
      @categories_words[category] = 0
    }
  end

  # train the document
  def train(category, document)
    word_count(document).each do |word, count|
      @words[category][word] ||= 0
      @words[category][word] += count
      @total_words += count
      @categories_words[category] += count
    end
    @categories_documents[category] += 1
    @total_documents += 1
  end

  # find the probabilities for each category and return a hash
  def probabilities(document)
    probabilities = Hash.new
    @words.each_key {|category| 
      probabilities[category] = probability(category, document)
    }
    return probabilities
  end

  # classfiy the document into one of the categories
  def classify(document, default='unknown')
    sorted = probabilities(document).sort {|a,b| a[1]<=>b[1]}
    best,second_best = sorted.pop, sorted.pop
    return best[0] if best[1]/second_best[1] > @threshold
    return default
  end

  def prettify_probabilities(document)
    probs = probabilities(document).sort {|a,b| a[1]<=>b[1]}
    totals = 0
    pretty = Hash.new
    probs.each { |prob| totals += prob[1]}
    probs.each { |prob| pretty[prob[0]] = "#{prob[1]/totals * 100}%"}
    return pretty
  end

  private

  # the probability of a word in this category
  # uses a weighted probability in order not to have zero probabilities
  def word_probability(category, word)
    (@words[category][word.stem].to_f + 1)/@categories_words[category].to_f
  end

  # the probability of a document in this category
  # this is just the cumulative multiplication of all the word probabilities for this category
  def doc_probability(category, document)
    doc_prob = 1
    word_count(document).each { |word| doc_prob *= word_probability(category, word[0]) }
    return doc_prob
  end

  # the probability of a category
  # this is the probability that any random document being in this category
  def category_probability(category)
    @categories_documents[category].to_f/@total_documents.to_f
  end

  # the un-normalized probability of that this document belongs to this category
  def probability(category, document)
    doc_probability(category, document) * category_probability(category)
  end

  # get a hash of the number of times a word appears in any document
  def word_count(document)
    words = document.gsub(/[^\w\s]/,"").split
    d = Hash.new
    words.each do |word|
      word.downcase! 
      key = word.stem
      unless COMMON_WORDS.include?(word) # remove common words
        d[key] ||= 0
        d[key] += 1
      end
    end
    return d
  end	


  COMMON_WORDS = ['a','able','about','above','abroad','according','accordingly','across','actually','adj','after','afterwards','again','against','ago','ahead','ain\'t','all','allow','allows','almost','alone','along','alongside','already','also','although','always','am','amid','amidst','among','amongst','an','and','another','any','anybody','anyhow','anyone','anything','anyway','anyways','anywhere','apart','appear','appreciate','appropriate','are','aren\'t','around','as','a\'s','aside','ask','asking','associated','at','available','away','awfully','b','back','backward','backwards','be','became','because','become','becomes','becoming','been','before','beforehand','begin','behind','being','believe','below','beside','besides','best','better','between','beyond','both','brief','but','by','c','came','can','cannot','cant','can\'t','caption','cause','causes','certain','certainly','changes','clearly','c\'mon','co','co.','com','come','comes','concerning','consequently','consider','considering','contain','containing','contains','corresponding','could','couldn\'t','course','c\'s','currently','d','dare','daren\'t','definitely','described','despite','did','didn\'t','different','directly','do','does','doesn\'t','doing','done','don\'t','down','downwards','during','e','each','edu','eg','eight','eighty','either','else','elsewhere','end','ending','enough','entirely','especially','et','etc','even','ever','evermore','every','everybody','everyone','everything','everywhere','ex','exactly','example','except','f','fairly','far','farther','few','fewer','fifth','first','five','followed','following','follows','for','forever','former','formerly','forth','forward','found','four','from','further','furthermore','g','get','gets','getting','given','gives','go','goes','going','gone','got','gotten','greetings','h','had','hadn\'t','half','happens','hardly','has','hasn\'t','have','haven\'t','having','he','he\'d','he\'ll','hello','help','hence','her','here','hereafter','hereby','herein','here\'s','hereupon','hers','herself','he\'s','hi','him','himself','his','hither','hopefully','how','howbeit','however','hundred','i','i\'d','ie','if','ignored','i\'ll','i\'m','immediate','in','inasmuch','inc','inc.','indeed','indicate','indicated','indicates','inner','inside','insofar','instead','into','inward','is','isn\'t','it','it\'d','it\'ll','its','it\'s','itself','i\'ve','j','just','k','keep','keeps','kept','know','known','knows','l','last','lately','later','latter','latterly','least','less','lest','let','let\'s','like','liked','likely','likewise','little','look','looking','looks','low','lower','ltd','m','made','mainly','make','makes','many','may','maybe','mayn\'t','me','mean','meantime','meanwhile','merely','might','mightn\'t','mine','minus','miss','more','moreover','most','mostly','mr','mrs','much','must','mustn\'t','my','myself','n','name','namely','nd','near','nearly','necessary','need','needn\'t','needs','neither','never','neverf','neverless','nevertheless','new','next','nine','ninety','no','nobody','non','none','nonetheless','noone','no-one','nor','normally','not','nothing','notwithstanding','novel','now','nowhere','o','obviously','of','off','often','oh','ok','okay','old','on','once','one','ones','one\'s','only','onto','opposite','or','other','others','otherwise','ought','oughtn\'t','our','ours','ourselves','out','outside','over','overall','own','p','particular','particularly','past','per','perhaps','placed','please','plus','possible','presumably','probably','provided','provides','q','que','quite','qv','r','rather','rd','re','really','reasonably','recent','recently','regarding','regardless','regards','relatively','respectively','right','round','s','said','same','saw','say','saying','says','second','secondly','see','seeing','seem','seemed','seeming','seems','seen','self','selves','sensible','sent','serious','seriously','seven','several','shall','shan\'t','she','she\'d','she\'ll','she\'s','should','shouldn\'t','since','six','so','some','somebody','someday','somehow','someone','something','sometime','sometimes','somewhat','somewhere','soon','sorry','specified','specify','specifying','still','sub','such','sup','sure','t','take','taken','taking','tell','tends','th','than','thank','thanks','thanx','that','that\'ll','thats','that\'s','that\'ve','the','their','theirs','them','themselves','then','thence','there','thereafter','thereby','there\'d','therefore','therein','there\'ll','there\'re','theres','there\'s','thereupon','there\'ve','these','they','they\'d','they\'ll','they\'re','they\'ve','thing','things','think','third','thirty','this','thorough','thoroughly','those','though','three','through','throughout','thru','thus','till','to','together','too','took','toward','towards','tried','tries','truly','try','trying','t\'s','twice','two','u','un','under','underneath','undoing','unfortunately','unless','unlike','unlikely','until','unto','up','upon','upwards','us','use','used','useful','uses','using','usually','v','value','various','versus','very','via','viz','vs','w','want','wants','was','wasn\'t','way','we','we\'d','welcome','well','we\'ll','went','were','we\'re','weren\'t','we\'ve','what','whatever','what\'ll','what\'s','what\'ve','when','whence','whenever','where','whereafter','whereas','whereby','wherein','where\'s','whereupon','wherever','whether','which','whichever','while','whilst','whither','who','who\'d','whoever','whole','who\'ll','whom','whomever','who\'s','whose','why','will','willing','wish','with','within','without','wonder','won\'t','would','wouldn\'t','x','y','yes','yet','you','you\'d','you\'ll','your','you\'re','yours','yourself','yourselves','you\'ve','z','zero']  
end
