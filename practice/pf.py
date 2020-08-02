import lyricsgenius

genius = lyricsgenius.Genius("VEohvWA5pH-fOGqWYSkR7Fnpd91uQKgjR_3SSgMXINKUelZNPu1Wc3JGwj_6BthA")
artist = genius.search_artist("Pink Floyd")

with open("pf.txt", "w") as file:
    for song in artist.songs:
        file.write(song.title + '\n\n\n')
        print(song.title)
        for line in song.lyrics.splitlines():
            if len(line) == 0 or line[0] != '[' or line[-1] != ']':
                file.write(line + '\n')
                print(line)