import heapq
import time
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import gensim.models.keyedvectors as word2vec


class Node:
    def __init__(self, elem, parent, priority):
        self.elem = elem
        self.parent = parent
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority


wikipedia_url = 'https://en.wikipedia.org'


def get_input():
    starting_title = input("Enter starting page name: ")
    starting_url = '/wiki/' + starting_title.replace(' ', '_')
    ending_title = input("Enter ending page name: ")
    ending_url = '/wiki/' + ending_title.replace(' ', '_')

    if not is_page_real(starting_url) or not is_page_real(ending_url):
        raise ValueError('One of the pages did not exist')

    return starting_url, ending_url


def is_link_good(link):
    if link['href'][:6] != '/wiki/':
        return False

    dumb_words = [
        'Wikipedia',
        'Special',
        'Help',
        'Main_Page',
        'Portal',
        'disambiguation',
        'File',
        'Category',
        'Talk',
        'Template',
        ':'
    ]

    for dumb_word in dumb_words:
        if dumb_word in link['href']:
            return False
    return True


def is_page_real(link):
    page_content = requests.get(wikipedia_url + link).content
    page_soup = BeautifulSoup(page_content, 'html5lib')
    page_body = page_soup.find('body').get_text()
    if 'Wikipedia does not have an article with this exact name' in page_body or 'may refer to' in page_body:
        return False
    return True


def get_all_links(link):
    page_content = requests.get(link).content
    link_soup = BeautifulSoup(page_content, 'html5lib')
    links = list(set([link['href'] for link in link_soup.find_all(
        'a', href=True) if is_link_good(link)]))
    return links


def rate_link_similarity_to_goal(model, query_link, goal_link):
    query_string = query_link[6:].replace('_', ' ')
    goal_string = goal_link[6:].replace('_', ' ')
    try:
        similarity = model.similarity(query_string, goal_string)
        return -1 * similarity
    except:
        return 0


def search_links(model, start, goal):
    goal_found = False
    frontier = [Node(start, None, 0)]
    visited = []
    current = None

    while not goal_found and len(frontier) > 0:
        current = heapq.heappop(frontier)
        if current.elem in visited:
            continue
        visited.append(current.elem)
        link = wikipedia_url + current.elem
        children = get_all_links(link)
        for kid in children:
            similarity = rate_link_similarity_to_goal(
                model, kid, goal)
            kid_node = Node(kid, current, similarity)
            if kid == goal:
                goal_found = True
                current = kid_node
                break
            else:
                heapq.heappush(frontier, kid_node)

    if not goal_found:
        return []
    else:
        path = []
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]


def _main():
    print('Loading...')
    model = word2vec.KeyedVectors.load_word2vec_format(
        '../pretrained-nlp-vector/GoogleNews-vectors-negative300.bin.gz', binary=True)
    keep_playing = True
    while keep_playing:
        try:
            starting_url, ending_title = get_input()
            start_time = datetime.now()
            path = search_links(model, starting_url, ending_title)
            end_time = datetime.now()
            elapsed_time = end_time - start_time
            formatted_time = elapsed_time.total_seconds()
            for step in path:
                print(step.elem)
            print(f'Elapsed time: {formatted_time} seconds')
            answer = input('Type Y to play again, anything else to quit: ')
            if answer != 'Y':
                keep_playing = False
        except:
            print(
                'An error occurred. Please ensure that the page names exist on Wikipedia.')
    print('Goodbye!')


if __name__ == "__main__":
    _main()
