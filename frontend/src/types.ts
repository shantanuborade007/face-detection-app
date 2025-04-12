export interface Song {
    id: string;
    name: string;
    artist: string;
    album: string;
    preview_url: string | null;
    external_url: string;
    album_image: string | null;
}

export interface FilterOptionsType {
    language: string;
    era: string;
    limit: number;
}