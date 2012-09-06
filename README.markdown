I first learnt about probability when I was in secondary school. As with all the other topics in Maths, it was just another bunch of formulas to memorize and regurgitate to apply to exam questions. Although I was curious if there was any use for it beyond calculating the odds for gambling, I didn't manage to find out any. As with many things in my life, things pop up at unexpected places and I stumbled on it again when as I started on machine learning and naive Bayesian classifiers.

A classifier is exactly that -- it's something that classifies other things. A classifier is a function that takes in a set of data and tells us which category or classification the data belongs to. A naive Bayesian classifier is a type of learning classifier, meaning that you can continually train it with more data and it will be be better at its job. The reason why it's called Bayesian is because it uses [Bayes Law](http://en.wikipedia.org/wiki/Bayes%27_theorem), a mathematical theorem that talks about conditional probabilities of events, to determine how to classify the data. The classifier is called 'naive' because it assumes each event (in this case the data) to be totally unrelated to each other. That's a very simplistic view but in practice it has been proven to be a surprisingly accurate. Also, because it's relatively simple to implement, it's quite popular. Amongst its more well-known usage include email spam filters.

So what's Bayes' Law and how can it be used to categorize data? As mentioned, Bayes' Law describes <a href="http://en.wikipedia.org/wiki/Conditional_probabilities" target="_blank">conditional probabilities</a>. An example of conditional probability is the probability of an event A happening given that another event B has happened. This is usually written as Pr(A | B), which is read as the probability of A, given B. To classify a document, we ask -- given a particular text document, what's the probability that it belongs to this category? When we find the probabilities of the given document in all categories, the classifier picks the category with the highest probability and announce it as the winner, that is, the document <strong>most probably</strong> belongs to that category.

The question then follows, how to we get the probability of a document belonging to a category? This is where we turn to Bayes' Law which states that:

<a rel="attachment wp-att-427" href="http://blog.saush.com/2009/02/11/naive-bayesian-classifiers-and-ruby/bayeslaweq11/"><img class="alignnone size-medium wp-image-427" title="bayeslaweq11" src="http://saush.wordpress.com/files/2009/02/bayeslaweq11.png?w=300" alt="bayeslaweq11" width="210" height="46" /></a>

Given our usage, what we want is:

<a href="http://blog.saush.com/wp-content/uploads/2009/02/bayeslaweq2.png"></a><a rel="attachment wp-att-428" href="http://blog.saush.com/?attachment_id=428"></a><a rel="attachment wp-att-429" href="http://blog.saush.com/2009/02/11/naive-bayesian-classifiers-and-ruby/bayeslaweq2-300x29/"><img class="alignnone size-full wp-image-429" title="bayeslaweq2-300x29" src="http://saush.wordpress.com/files/2009/02/bayeslaweq2-300x29.png" alt="bayeslaweq2-300x29" width="300" height="29" /></a>

What we need is Pr(document|category) and Pr(category). You should keep in mind that we're comparing relative probabilities here so we can drop Pr(document) because it is the same for every category.

What is Pr(document|category) and how do we find it? It is the probability that this document exists, given a particular category. As a document is made of a bunch of words, what we need to do is to calculate the probability of the words in the document within the category. Here is where the 'naive' part comes in. We know that the words in a document are not random and the probability of a word like 'Ruby' would be more likely to be found in an article on the Ruby programming language than say, an article on the dental practices in Uganda. However for the purpose of simplicity, the naive Bayesian classifier treats each word as independent of each other.

Remember your probability lessons -- if the probability of each word is independent of each other, the probability of a whole bunch of words together is the product of the probability of each word in the bunch. A quick aside to illustrate this.

Take a pair of dice and roll them one after another. The probability of the the first die to fall on any one of its 6 sides is 1 out of 6, that is 1/6. The probability of the second die to fall on any one of its 6 sides is also 1/6. So what is the probability that both dice lands on 6? Out of the 6 x 6 = 36 possible ways that a pair of dice can land there is only 1 way that both dice lands on 6, so the probability is 1/36, which is 1/6 x 1/6. This is true only if the dice rolls are independent of each other. In the same way we are 'naively' assuming that the words in the document are occurring independently of each other, as if it is written by the theoretical monkey with a typewriter.

In other words, the probability that a document exists, given a category, is the product of the probability of each word in that document. Now that we've established this, how do we get the probability of a single word? Basically it's the count of the number of times the word appeared in the category after the classifier has been trained compared to the total word counts in that category. Another quick illustration. Say we train the classifier with 2 categories (spam and not-spam), there are 100 word counts in the spam category. There are only be 14 unique words in this category but some of these words have been trained more than once. Out of these 100 word counts, 5 of them are for the word 'money'. The probability for the word 'money' would be the number of times it is mentioned in the spam category (5) divided by the number of word counts in this category (100).

Now that we know Pr(document|category) let's look at Pr(category). This is simply the probability of any document being in this category (instead of being in another category). This is the number of documents used to train this category over the total number of documents that used to train all categories.

So that's the basic idea behind naive Bayesian classifiers. With that I'm going to show you how to write a simple classifier in Ruby. There is already a rather popular Ruby implementation by Lucas Carlsson called the <a href="http://classifier.rubyforge.org" target="_blank">Classifier gem (http://classifier.rubyforge.org)</a> which you can use readily but let's write our own classifier instead. We'll be creating class named <em>NativeBayes</em>, in a file called <em>native_bayes.rb</em>. This classifier will be used to classify text into different categories. Let's recap how this classifier will be used:

* First, tell the classifier how many categories there will be
* Next, train the classifier with a number of documents, while indicating which category those document belongs to
* Finally, pass the classifier a document and it should tell us which category it thinks the document should be in

Now let's run through the public methods of the *NativeBayes* class, which should map to the 3 actions above:

* Provide the categories you want to classify the data into
* Train the classifier by feeding it data
* Doing the real work, that is to classify given data

The first method we'll roll into the constructor of the class, so that when we create the object, the categories will be set. The second method, <em>train</em>, takes in a category and a document (a text string) to train the classifier. The last method, <em>classify</em>, takes in just a document (a text string) and returns its category.

    class NaiveBayes
      def initialize(*categories)
        @words = Hash.new
        @total_words = 0
        @categories_documents = Hash.new
        @total_documents = 0
        @categories_words = Hash.new
        @threshold = 1.5

        categories.each { |category|
          @words[category] = Hash.new
          @categories_documents[category] = 0
          @categories_words[category] = 0
        }
      end

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

      def classify(document, default='unknown')
        sorted = probabilities(document).sort {|a,b| a[1]<=>b[1]}
        best,second_best = sorted.pop, sorted.pop
        return best[0] if (best[1]/second_best[1] > @threshold)
        return default
      end
    end

Let's look at the initializer first. We'll need the following instance variables:
1. @words is a hash containing a list of words trained for the classifier. It looks something like this:

    {
    "spam"    => { "money" => 10, "quick" => 12, "rich"  => 15 },
    "not_spam" => { "report" => 9, "database" => 13, "salaries"  => 12 }
    }

"Spam" and "not_spam" are the categories, while "money", "quick" etc are the words in the "spam" category with the numbers indicating the number of times it has been trained as that particular category.  

2. @total_words contains the number of words trained  

3. @categories_documents is a hash containing the number of documents trained for each category:

    { "spam" => 4, "not_spam" => 5}

4. @total_documents is the total number of documents trained  

5. @categories_words is a hash containing the number of words trained for each category:

    { "spam" => 37, "not_spam" => 34}

6. @threshold is something I will talk about again at the last section of the code descriptions (it doesn't make much sense now).  Next is the <em>train</em> method, which takes in a category and a document. We break down the document into a number of words, and slot it accordingly into the instance variables we created earlier on. Here we are using a private helper method called word_count to do the grunt work.

    def word_count(document)
      words = document.gsub(/[^\w\s]/,"").split
      d = Hash.new
      words.each do |word|
        word.downcase!
        key = word.stem
        unless COMMON_WORDS.include?(word)
        d[key] ||= 0
        d[key] += 1
      end
    end
    return d
    end

    COMMON_WORDS = ['a','able','about','above','abroad' ...] # this is truncated

The code is quite straightforward, we're just breaking down a text string into its constituent words. We want to focus on words that characterize the document so we're really not that interested in some words such as pronouns, conjunctions, articles, and so on. Dropping those common words will bring up nouns, characteristic adjectives and some verbs. Also, to reduce the number of words, we use a technique called 'stemming' which essentially reduces any word to its 'stem' or root word. For example, the words 'fishing', 'fisher', 'fished', 'fishy' are all reduced to a root word 'fish'. In our method here we used a popular stemming algorithm called <a href="http://blog.saush.com/2009/02/word-stemming-in-ruby/" target="_self">Porter stemming algorithm</a>. To use this stemming algorithm, install the following gem:

    gem install stemmer

Now let's look at the *classify* method. This is the method that uses <a href="http://en.wikipedia.org/wiki/Bayes%27_theorem" target="_blank">Bayes' Law</a> to classify documents. We will be breaking it down into a few helper methods to illustrate how Bayes' Law is used. Remember that finally we're looking at the probability of a given document being in any of the categories, so we need to have a method that returns a hash of categories with their respective probabilities like this:

<code>
{ "spam" => 0.123, "not_spam" => 0.327}
</code>

    def probabilities(document)
      probabilities = Hash.new
      @words.each_key {|category|
        probabilities[category] = probability(category, document)
      }
      return probabilities
    end

In the *probabilities* method, we need to calculate the probability of that document being in each category. As mentioned above, that probability is Pr(document|category) * Pr(category). We create a helper method called *probability* that simply multiplies the document probability Pr(document|category) and the category probability Pr(category).

    def probability(category, document)
      doc_probability(category, document) * category_probability(category)
    end

First let's tackle Pr(document|category). To do that we need to get all the words in the given document, get the word probability of that document and multiply them all together.

    def doc_probability(category, document)
      doc_prob = 1 
      word_count(document).each { |word| doc_prob *= word_probability(category, word[0]) } 
      return doc_prob
    end

Next, we want to get the probability of a word. Basically the probability of a word in a category is the number of times it occurred in that category, divided by the number of words in that category altogether. However if the word never occurred during training (and this happens pretty frequently if you don't have much training data), then what you'll get is a big fat 0 in probability. If we propagate this upwards, you'll notice that the document probability will all be made 0 and therefore the probability of that document in that category is made 0 as well. This, of course, is not the desired results. To correct it, we need to tweak the formula a bit.
To make sure that there is at least some probability to the word even if it isn't in trained list, we assume that the word exists at least 1 time in the training data so that the result is not 0. So this means that instead of:
<a href="http://blog.saush.com/wp-content/uploads/2009/02/word_prob1_eq.png"></a><a rel="attachment wp-att-430" href="http://blog.saush.com/2009/02/11/naive-bayesian-classifiers-and-ruby/word_prob1_eq-300x47/"><img class="alignnone size-full wp-image-430" title="word_prob1_eq-300x47" src="http://saush.wordpress.com/files/2009/02/word_prob1_eq-300x47.png" alt="word_prob1_eq-300x47" width="300" height="47" /></a>
we take:
<a href="http://blog.saush.com/wp-content/uploads/2009/02/word_prob2_eq.png"></a><a rel="attachment wp-att-431" href="http://blog.saush.com/2009/02/11/naive-bayesian-classifiers-and-ruby/word_prob2_eq-300x44/"><img class="alignnone size-full wp-image-431" title="word_prob2_eq-300x44" src="http://saush.wordpress.com/files/2009/02/word_prob2_eq-300x44.png" alt="word_prob2_eq-300x44" width="300" height="44" /></a>
So the code is something like this:

    def word_probability(category, word)
     (@words[category][word.stem].to_f + 1)/@categories_words[category].to_f
    end

Finally we want to get Pr(category), which is pretty straightforward. It's just the probability that any random document being in this category, so we take number of documents trained in the category and divide it with the total number of documents trained in the classifier.

    def category_probability(category)
      @categories_documents[category].to_f/@total_documents.to_f
    end

Now that we have the probabilities, let's go back to the classify method and take a look at it again:

     def classify(document, default='unknown')
       sorted = probabilities(document).sort {|a,b| a[1]<=>b[1]}
       best,second_best = sorted.pop, sorted.pop
       return best[0] if (best[1]/second_best[1] > @threshold)
       return default
     end

We sort the probabilities to bubble up the category with the largest probability. However if we use this directly,it means it has to be the one with the largest, even though the category with the second largest probability is only maybe a bit smaller. For example, take the spam and non-spam categories and say the ratio of the probabilities are like this -- spam is 53% and non-spam is 47%. Should the document be classified as spam? Logically, not! This is the reason for the threshold variable, which gives a ratio between the best and the second best. In the example code above the value is 1.5 meaning the best probability needs to be 1.5 times better than the second best probability i.e. the ratio needs to be 60% to 40% for the best and second best probabilities respectively. If this is not the case, then the classifier will just shrug and say it doesn't know (returns 'default' as the category). You can tweak this number accordingly depending on the type of categories you are using.
Now that we have the classifier, let's take it out for a test run. I'm going to use a set of Yahoo news RSS feeds to train the classifier according to the various categories, then use some random text I get from some other sites and ask the classifier to classify them.

    require 'rubygems'
    require 'rss/1.0'
    require 'rss/2.0'
    require 'open-uri'
    require 'hpricot'
    require 'naive_bayes'
    require 'pp'

    categories = %w(tech sports business entertainment)
    classifier = NaiveBayes.new(categories)

    content =''
    categories.each { |category|
      feed = "http://rss.news.yahoo.com/rss/#{category}"
      open(feed) do |s| content = s.read end
      rss = RSS::Parser.parse(content, false)
      rss.items.each { |item|
      text = Hpricot(item.description).inner_text
      classifier.train(category, text)
    }

    # classify this
    documents = [
    "Google said on Monday it was releasing a beta version of Google Sync for the iPhone and Windows Mobile phones",
    :Rangers waste 5 power plays in 3-0 loss to Devils",
    "Going well beyond its current Windows Mobile software, Microsoft will try to extend its desktop dominance with a Windows phone.",
    "UBS cuts jobs after Q4 loss",
    "A fight in Hancock Park after a pre-Grammy Awards party left the singer with bruises and a scratched face, police say."]

    documents.each { |text|
      puts text
      puts "category => #{classifier.classify(text)}"
      puts
    }


This is the output:

    Google said on Monday it was releasing a beta version of Google Sync for the iPhone and Windows Mobile phones
    {"tech"=>"62.4965904628081%",
     "business"=>"6.51256988628851%",
     "entertainment"=>"16.8319433691552%",
     "sports"=>"14.1588962817482%"}
    category => tech

    Rangers waste 5 power plays in 3-0 loss to Devils
    {"tech"=>"14.7517595939031%",
     "business"=>"3.51842781998617%",
     "entertainment"=>"8.09457974962582%",
     "sports"=>"73.6352328364849%"}
    category => sports

    Going well beyond its current Windows Mobile software, Microsoft will try to extend its desktop dominance with a Windows phone.
    {"tech"=>"91.678065974899%",
     "business"=>"0.851666657161468%",
     "entertainment"=>"4.15349223570253%",
     "sports"=>"3.31677513223704%"}
    category => tech

    UBS cuts jobs after Q4 loss
    {"business"=>"33.1048977545484%",
     "tech"=>"14.1719403525702%",
     "entertainment"=>"31.9818519810561%",
     "sports"=>"20.7413099118253%"}
    category => unknown

    A fight in Hancock Park after a pre-Grammy Awards party left the R&B singer with bruises and a scratched face, police say.
    {"tech"=>"4.10704270254326%",
     "business"=>"1.4959136651331%",
     "entertainment"=>"78.1802587499558%",
     "sports"=>"16.2167848823678%"}
    category => entertainment

You can download the code described above from GitHub at <a href="http://github.com/sausheong/naive-bayes" target="_blank">http://github.com/sausheong/naive-bayes</a>, including naive_bayes.rb and the bayes_test.rb files.
For more information, you should pick up the excellent book <a href="http://oreilly.com/catalog/9780596529321/" target="_blank">Programming Collective Intelligence</a> by <a href="http://blog.kiwitobes.com/" target="_blank">Toby Segaran</a>.
