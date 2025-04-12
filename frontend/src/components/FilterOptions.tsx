import React from 'react';
import { FilterOptionsType } from '../types';

interface FilterOptionsProps {
  filters: FilterOptionsType;
  onFilterChange: (filters: FilterOptionsType) => void;
}

const FilterOptions: React.FC<FilterOptionsProps> = ({ filters, onFilterChange }) => {
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const { name, value } = e.target;
    
    onFilterChange({
      ...filters,
      [name]: value
    });
  };
  
  return (
    <div className="filter-options">
      <h2>Song Preferences</h2>
      
      <div className="filters-container">
        <div className="filter-group">
          <label htmlFor="language">Language</label>
          <select 
            name="language" 
            id="language" 
            value={filters.language}
            onChange={handleChange}
          >
            <option value="">Any Language</option>
            <option value="english">English</option>
            <option value="hindi">Hindi</option>
            <option value="spanish">Spanish</option>
            <option value="french">French</option>
            <option value="korean">Korean</option>
            <option value="japanese">Japanese</option>
          </select>
        </div>
        
        <div className="filter-group">
          <label htmlFor="era">Era</label>
          <select 
            name="era" 
            id="era" 
            value={filters.era}
            onChange={handleChange}
          >
            <option value="">Any Era</option>
            <option value="1960s">1960s</option>
            <option value="1970s">1970s</option>
            <option value="1980s">1980s</option>
            <option value="1990s">1990s</option>
            <option value="2000s">2000s</option>
            <option value="2010s">2010s</option>
            <option value="2020s">2020s</option>
          </select>
        </div>
        
        <div className="filter-group">
          <label htmlFor="limit">Number of Songs</label>
          <select 
            name="limit" 
            id="limit" 
            value={filters.limit.toString()}
            onChange={handleChange}
          >
            <option value="5">5</option>
            <option value="10">10</option>
            <option value="15">15</option>
            <option value="20">20</option>
          </select>
        </div>
      </div>
    </div>
  );
};

export default FilterOptions;