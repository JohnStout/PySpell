PySpell

A collection of codes used to convert, store, and process imaging data.

Note that if you are attempting to use recurseConvert, you can store your data anywhich way you choose.

If you are attempting to use synchronizeToDropbox, you should store your data as such:
    if you have a recording "recording_1" 
    AND
        `fpath = path/to/recording_1`,
    THEN
        `bpath = path/to/recording_1/recording1_beh`
    AND 
        `imgpath = path/to/recording_1/recording1_img`

code/├── s2pfuns


------

Note that if working over dropbox environments, please run the following commands in anaconda prompt:


`conda config --add envs_dirs "path\to.....\SpellmanLab Dropbox\timspellman\Python\envs"`

then you can

`conda activate dpsuite2p`