#include <iostream>
#include <fstream> 
#include <string>
#include <vector>
#include <charconv>
#include <string_view>
#include <ranges>
#include <queue>
#include "./../utilities/graph_io.h"
using namespace std;

//#343fff getting the file access and parsing once
int file_access(Graph& adj_list){
    string cache_path = "/home/fakeheadset/Projects/EulerEasel/src/SerialBaseline";
    ifstream check_cache(cache_path, ios::binary);
    if(check_cache.good()){
        check_cache.close();
        cout<<"Cache found"<<endl;
        load_graph_structure(adj_list, cache_path);
        return 0;
    }
    check_cache.close();
    cout<<"No cache file found"<<endl;
    ifstream inputFile("/home/fakeheadset/Projects/EulerEasel/Data/graph.txt");
    
    if(!inputFile.is_open()){
        cerr<<"this file couldn't open"<<endl;
        return -1;
    }

    string dummyline;
    for(int i=0;i<5;i++){
        if(!getline(inputFile, dummyline)) return -1;
    }

    string line;
    vector<int> numbers;
    while(getline(inputFile, line)){
        //split the string
        auto words = views::split(line, '\t') | views::filter([](auto && subrange) {return !subrange.empty();}); 

        for( auto const word: words){
            string_view token(word.begin(), word.end());

            if(!token.empty() && token.back() == '\r'){
                //removes the '/r' and the len-=1
                token.remove_suffix(1);
            }

            if (token.empty()) continue;
            int value=0;
            //conversion to string to numbers
            auto [ptr, ec] = from_chars(token.data(), token.data() + token.size(), value);

            if(ec == errc()) numbers.push_back(value);
        }   

        if (numbers.size()>=2){
            adj_list.push_back(numbers);
        }
        
        numbers.clear();
    }   

    inputFile.close();

    cout<<"saving graph structure to binary cache for future runs"<<endl;
    save_graph_struture(adj_list, cache_path);

    return 0;
}